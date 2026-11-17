"""Persist reproducible filter decisions without removing generated artifacts."""
from dataclasses import asdict
from datetime import datetime,timezone
from pathlib import Path
from uuid import uuid4
import hashlib,json
from .foreground import object_mask
from .measurements import measure
from .rules import violations
from .reports import write_csv,preview

def config_fingerprint(config):
    return hashlib.sha256(json.dumps(config.to_dict(),sort_keys=True).encode()).hexdigest()

def artifact_signature(hybrid,original,store):
    values=[asdict(hybrid),asdict(original)]
    for path in [hybrid.image_path,hybrid.segmentation_path,original.image_path]:
        stat=store.resolve(path).stat();values.append((path,stat.st_size,stat.st_mtime_ns))
    return hashlib.sha256(json.dumps(values,sort_keys=True).encode()).hexdigest()

def decision_for(repository,store,hybrid,original,config):
    """Require current decisions whenever the caller explicitly enables filtering."""
    if not config.enabled:return None
    with repository.connection() as db:
        row=db.execute('select * from postprocessing_decisions where hybrid_sample_id=?',(hybrid.id,)).fetchone()
    if (row is None or row['config_fingerprint']!=config_fingerprint(config)
            or row['artifact_signature']!=artifact_signature(hybrid,original,store)):
        raise ValueError('Postprocessing decisions missing or stale. Run postprocess_hybrid_samples() again.')
    return dict(row)

class PostprocessingService:
    def __init__(self,repository,store,config):
        self.repository=repository;self.store=store;self.config=config

    def run(self):
        config=self.config;config.validate()
        if not config.enabled:return {'enabled':False}
        hybrids=self.repository.list_hybrid_samples()
        if not hybrids or any(h.status!='generated' for h in hybrids):
            raise ValueError('Postprocessing requires a completely materialized hybrid plan.')
        run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid4().hex[:8]
        folder=self.store.study_folder/'postprocessing'/run_id;folder.mkdir(parents=True)
        if config.save_previews:(folder/'previews').mkdir()
        (folder/'configuration.json').write_text(json.dumps(config.to_dict(),indent=2)+'\n')
        fingerprint=config_fingerprint(config);decisions=[];rows=[];rejected=0
        # Process siblings together, retaining only one original/object mask in RAM.
        current=None;image=foreground=None
        for hybrid in sorted(hybrids,key=lambda h:(h.original_sample_id,h.id)):
            original=self.repository.get_original_sample(hybrid.original_sample_id)
            if original.has_anomaly and config.rules.minimum_object_overlap.enabled:
                raise ValueError('Object overlap requires healthy original partners, not anomalous source images.')
            if current!=original.id:
                image=self.store.load_array(original.image_path,mmap_mode='r')
                foreground=object_mask(image,config.foreground) if config.rules.minimum_object_overlap.enabled else None
                current=original.id
            mask=self.store.load_array(hybrid.segmentation_path,mmap_mode='r')
            values=measure(mask,foreground);reasons=violations(values,config.rules);accepted=not reasons
            rejected+=not accepted
            decisions.append((hybrid.id,run_id,fingerprint,artifact_signature(hybrid,original,self.store),int(accepted),json.dumps(values),json.dumps(reasons)))
            rows.append({'hybrid_id':hybrid.id,'accepted':accepted,'reasons':json.dumps(reasons),'measurements':json.dumps(values)})
            if config.save_previews:preview(folder/'previews'/f'{hybrid.id}.png',image,mask,foreground)
        summary={'enabled':True,'run_id':run_id,'action':config.action,'total':len(hybrids),'passed':len(hybrids)-rejected,'failed':rejected,'excluded':rejected if config.action=='exclude' else 0,'report_folder':str(folder),'config_fingerprint':fingerprint}
        write_csv(folder/'decisions.csv',rows)
        (folder/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        # Publish only after all images and reports have been processed successfully.
        with self.repository.connection() as db:
            db.execute('delete from postprocessing_decisions')
            db.executemany('insert into postprocessing_decisions values (?,?,?,?,?,?,?)',decisions)
        return summary
