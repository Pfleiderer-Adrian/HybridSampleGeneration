
# Keep explicit frozen PYTHONPATH entries ahead of the live checkout.
import sys as _analysis_sys
from pathlib import Path as _AnalysisPath
if __package__ in (None, ""):
    _analysis_sys.path.append(str(_AnalysisPath(__file__).resolve().parents[4]))
from scripts.analysis.paths import audit_dir

from pathlib import Path
import sqlite3,json
import numpy as np
s=Path('/mnt/results/mvtec2/experiments/comparison_v3/wallplugs/hybrid_generation');db=sqlite3.connect(s/'artifacts.sqlite');db.row_factory=sqlite3.Row;out=audit_dir('wallplugs_v3_analysis');groups={}
for kind,query in [('hybrid','select h.id,h.segmentation_path,o.image_path,o.source_name from hybrid_samples h join original_samples o on o.id=h.original_sample_id'),('real','select id,segmentation_path,image_path,source_name from original_samples where has_anomaly=1')]:
 rows=[]
 for r in db.execute(query):
  a=np.load(s/r['image_path'],mmap_mode='r')[0];mask=(np.load(s/r['segmentation_path'],mmap_mode='r')>0).any(0);values=a[mask];n=len(values)
  rows.append({'id':r['id'],'target':r['source_name'],'mask_pixels':n,**{f'dark_{t}':float((values<t).mean()) if n else 0 for t in [20,30,40]}})
 groups[kind]=rows
 print(kind,'count',len(rows),'areas p10p50p90max',np.percentile([r['mask_pixels'] for r in rows],[10,50,90,100]).tolist())
 for t in [20,30,40]:print('threshold',t,'median fraction',float(np.median([r[f'dark_{t}'] for r in rows])),'>10%',sum(r[f'dark_{t}']>.1 for r in rows),'>50%',sum(r[f'dark_{t}']>.5 for r in rows))
 print('largest',sorted(rows,key=lambda r:r['mask_pixels'],reverse=True)[:4])
(out/'mask_measurements.json').write_text(json.dumps(groups,indent=2)+'\n')
