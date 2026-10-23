"""Persist reproducible filter decisions without removing generated artifacts."""
from dataclasses import asdict
from datetime import datetime,timezone
from pathlib import Path
from uuid import uuid4
import hashlib,json,re
import numpy as np
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

def source_group(name):
    """Capture identity shared by regular/exposure/shift views."""
    return re.sub(r'_(?:regular|overexposed|underexposed|shift_\d+)\.[^.]+$', '', name)

def overlap_mask(repository,store,hybrid,final_mask,exempt_groups):
    """Only nonexempt placements contribute to object-overlap checking."""
    result=np.zeros_like(final_mask,dtype=bool);exempt=0
    for placement in repository.list_placements(hybrid.id):
        synthetic=repository.get_synthetic_anomaly(placement.synthetic_anomaly_id)
        real=repository.get_real_anomaly(synthetic.real_anomaly_id)
        original=repository.get_original_sample(real.original_sample_id)
        if source_group(original.source_name) in exempt_groups:
            exempt+=1;continue
        path=store.relative_path('placements',placement.id,'full_segmentation')
        if not store.resolve(path).exists():
            raise ValueError('Source-group exemptions require per-placement full masks. Rematerialize this study.')
        value=store.load_array(path,mmap_mode='r')
        if value.shape!=final_mask.shape:raise ValueError('Placement mask shape mismatch.')
        result |= value>0
    return result & (final_mask>0),exempt

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
            values=measure(mask,foreground)
            groups=config.rules.minimum_object_overlap.exempt_source_groups
            if foreground is not None and groups:
                checked,exempt=overlap_mask(self.repository,self.store,hybrid,mask,groups)
                checked_values=measure(checked,foreground)
                values.update(object_overlap=checked_values['object_overlap'] if checked_values['mask_pixels'] else 1.,
                              outside_object_pixels=checked_values['outside_object_pixels'],
                              overlap_checked_pixels=checked_values['mask_pixels'],exempt_placements=exempt)
            reasons=violations(values,config.rules);accepted=not reasons
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
