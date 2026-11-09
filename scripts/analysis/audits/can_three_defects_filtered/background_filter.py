"""Can-specific, reproducible foreground heuristic; not a learned object mask."""

# Keep explicit frozen PYTHONPATH entries ahead of the live checkout.
import sys as _analysis_sys
from pathlib import Path as _AnalysisPath
if __package__ in (None, ""):
    _analysis_sys.path.append(str(_AnalysisPath(__file__).resolve().parents[4]))

import json,sqlite3,shutil
from pathlib import Path
import numpy as np
from scipy.ndimage import label,binary_fill_holes

def object_mask(original):
    # White external background is uniform; keep the largest nonwhite object.
    nonwhite=np.min(original,axis=0)<240
    labels,n=label(nonwhite)
    if not n:raise ValueError('No object detected')
    counts=np.bincount(labels.ravel());counts[0]=0
    result=binary_fill_holes(labels==counts.argmax())
    fraction=float(result.mean())
    if not .15<fraction<.95:raise ValueError(f'Unexpected Can object fraction: {fraction}')
    return result

def filter_study(study):
    study=Path(study);db=sqlite3.connect(study/'artifacts.sqlite');db.row_factory=sqlite3.Row
    db.execute('pragma foreign_keys=on')
    rows=db.execute('select h.id,h.original_sample_id,h.segmentation_path,o.image_path,o.source_name from hybrid_samples h join original_samples o on o.id=h.original_sample_id order by h.original_sample_id,h.id').fetchall()
    audit=[];previous=None;obj=None
    for row in rows:
        if previous!=row['original_sample_id']:
            obj=object_mask(np.load(study/row['image_path'],mmap_mode='r'));previous=row['original_sample_id']
        mask=np.load(study/row['segmentation_path'],mmap_mode='r')>0
        if mask.ndim==3:mask=mask.any(0)
        pixels=int(mask.sum());outside=int(np.count_nonzero(mask&~obj))
        audit.append({'hybrid_id':row['id'],'target':row['source_name'],'mask_pixels':pixels,'outside_object_pixels':outside,'outside_fraction':outside/max(1,pixels),'remove':outside>0 or pixels==0})
    removed=[r['hybrid_id'] for r in audit if r['remove']]
    report={'method':'Largest nonwhite connected component of healthy original, min(R,G,B)<240; fill internal holes; remove whole hybrid if any segmentation pixel lies outside. Can-specific heuristic, not semantic ground truth.','before':len(rows),'removed':len(removed),'retained':len(rows)-len(removed),'records':audit}
    (study/'background_filter_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    if not report['retained']:raise ValueError('Background filter removed every hybrid; stopping before deletion')
    # Only derived artifacts in this new study are removed. Original/real arrays
    # and generator checkpoints are never changed.
    placements=[]
    for hid in removed:
        placements.extend(r[0] for r in db.execute('select id from placements where hybrid_sample_id=?',(hid,)))
    with db:
        for hid in removed:db.execute('delete from hybrid_samples where id=?',(hid,))
    assert not db.execute('pragma foreign_key_check').fetchall()
    db.close()
    for entity,ids in [('hybrid_samples',removed),('placements',placements)]:
        for entity_id in ids:
            path=study/'artifacts'/entity/entity_id
            assert path.parent==study/'artifacts'/entity and not path.is_symlink()
            if path.exists():shutil.rmtree(path)
    return {k:v for k,v in report.items() if k!='records'}
