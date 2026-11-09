
# Keep explicit frozen PYTHONPATH entries ahead of the live checkout.
import sys as _analysis_sys
from pathlib import Path as _AnalysisPath
if __package__ in (None, ""):
    _analysis_sys.path.append(str(_AnalysisPath(__file__).resolve().parents[4]))
from scripts.analysis.paths import audit_dir

from pathlib import Path
import json,sqlite3,numpy as np
from scipy.ndimage import label
s=Path('/mnt/results/mvtec2/experiments/comparison_v3/fruit_jelly/hybrid_generation');out=audit_dir('fruit_jelly_v3_analysis');c=sqlite3.connect(f'file:{s}/artifacts.sqlite?mode=ro',uri=True);c.row_factory=sqlite3.Row;rows=[]
for r in c.execute('select h.id,h.segmentation_path,o.image_path from hybrid_samples h join original_samples o on o.id=h.original_sample_id'):
 a=np.load(s/r['image_path']);m=(np.load(s/r['segmentation_path'])>0).any(0);white=(a.min(0)>=240);labs,n=label(white);edges=np.unique(np.concatenate([labs[0],labs[-1],labs[:,0],labs[:,-1]]));edges=edges[edges!=0];bg=np.isin(labs,edges);rows.append({'id':r['id'],'white_background_fraction':float(bg[m].mean())})
d={'heuristic':'Original RGB minimum >=240, components connected to image border; reflections/transparent areas may still be ambiguous','hybrids':len(rows),'any_background':sum(r['white_background_fraction']>0 for r in rows),'over_10_percent':sum(r['white_background_fraction']>.1 for r in rows),'over_50_percent':sum(r['white_background_fraction']>.5 for r in rows),'rows':rows};(out/'background.json').write_text(json.dumps(d,indent=2));print({k:v for k,v in d.items() if k!='rows'})
