"""Automatic read-only comparison after Rice materialization."""

# Keep explicit frozen PYTHONPATH entries ahead of the live checkout.
import sys as _analysis_sys
from pathlib import Path as _AnalysisPath
if __package__ in (None, ""):
    _analysis_sys.path.append(str(_AnalysisPath(__file__).resolve().parents[4]))
from scripts.analysis.paths import audit_dir

import sys,json,sqlite3,re,collections
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
new=Path(sys.argv[1]);old=Path(json.loads((new/'run_provenance.json').read_text())['source_study']);out=audit_dir('rice_v3_analysis');out.mkdir(exist_ok=True);summaries={};catalog={}
def group(name):return re.sub(r'_(regular|overexposed|underexposed|shift_\d+)\.png$','',name)
def rgb(a):return np.moveaxis(np.repeat(a,3,0) if a.shape[0]==1 else a,0,-1).clip(0,255).astype('uint8')
for name,study in [('vorher',old),('neu',new)]:
 db=sqlite3.connect(f'file:{study}/artifacts.sqlite?mode=ro',uri=True);db.row_factory=sqlite3.Row
 tables={t:{r['id']:dict(r) for r in db.execute('select * from '+t)} for t in ['original_samples','real_anomalies','synthetic_anomalies','hybrid_samples','placements']};ps=collections.defaultdict(list)
 for p in tables['placements'].values():ps[p['hybrid_sample_id']].append(p)
 hybrids=[h for h in tables['hybrid_samples'].values() if h['status']=='generated'];groups=collections.Counter();sources=collections.Counter();variants=collections.Counter();dists=[]
 for p in tables['placements'].values():
  syn=tables['synthetic_anomalies'][p['synthetic_anomaly_id']];r=tables['real_anomalies'][syn['real_anomaly_id']];donor=tables['original_samples'][r['original_sample_id']];groups[group(donor['source_name'])]+=1;sources[r['id']]+=1;variants[syn['id']]+=1;dists.append(float(np.hypot(p['position_y']-r['position_y'],p['position_x']-r['position_x'])))
 areas=[];contrasts=[];outside=[];bytarget=collections.defaultdict(list)
 for h in hybrids:
  orig=tables['original_samples'][h['original_sample_id']];bytarget[orig['source_name']].append(h)
  a=np.load(study/orig['image_path'],mmap_mode='r');b=np.load(study/h['image_path'],mmap_mode='r');mask=(np.load(study/h['segmentation_path'],mmap_mode='r')>0).any(0);delta=np.abs(b.astype('float32')-a.astype('float32')).max(0);areas.append(int(mask.sum()));contrasts.append(float(np.median(delta[mask])) if mask.any() else 0.);outside.append(int(((delta>1)&~mask).sum()))
 available=collections.Counter(group(tables['original_samples'][r['original_sample_id']]['source_name']) for r in tables['real_anomalies'].values())
 summaries[name]={'hybrids':len(hybrids),'placements':len(tables['placements']),'real_rois_available':len(tables['real_anomalies']),'real_rois_used':len(sources),'synthetic_variants_available':len(tables['synthetic_anomalies']),'synthetic_variants_used':len(variants),'capture_groups_used':len(groups),'capture_group_placements':dict(groups),'missing_capture_groups':sorted(set(available)-set(groups)),'median_mask_pixels':float(np.median(areas)),'median_inside_change':float(np.median(contrasts)),'hybrids_changed_outside_mask_gt1':sum(v>0 for v in outside),'median_normalized_source_distance':float(np.median(dists))}
 catalog[name]=(study,tables,bytarget)
 print('ANALYZED',name,summaries[name],flush=True)
common=sorted(set(catalog['vorher'][2])&set(catalog['neu'][2]));rng=np.random.default_rng(42);names=[common[i] for i in rng.choice(len(common),min(12,len(common)),replace=False)]
for page in range((len(names)+2)//3):
 canvas=Image.new('RGB',(1600,990),'white');draw=ImageDraw.Draw(canvas)
 for k,target in enumerate(names[page*3:page*3+3]):
  draw.text((10,k*330+5),target+' | old original+mask / new original+mask / old hybrid / new hybrid',fill='black');items=[]
  for key in ['vorher','neu']:
   study,tables,bytarget=catalog[key];h=min(bytarget[target],key=lambda h:h['variant_index']);orig=tables['original_samples'][h['original_sample_id']];a=rgb(np.load(study/orig['image_path']));b=rgb(np.load(study/h['image_path']));m=(np.load(study/h['segmentation_path'])>0).any(0);ov=a.copy();ov[m]=(ov[m]*.25+np.array([255,0,0])*.75).astype('uint8');y,x=np.where(m);cx=int(np.median(x));cy=int(np.median(y));box=(max(0,cx-180),max(0,cy-140),min(m.shape[1],cx+180),min(m.shape[0],cy+140));items.append((Image.fromarray(ov).crop(box),Image.fromarray(b).crop(box)))
  for j,im in enumerate([items[0][0],items[1][0],items[0][1],items[1][1]]):im.thumbnail((390,290));canvas.paste(im,(j*400,k*330+30))
 canvas.save(out/f'paired_{page+1}.jpg',quality=94)
(out/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
lines=['# Rice: Vergleich nach neuer Generierung','', 'Automatische quantitative Auswertung; Bildtafeln fuer anschliessende Sichtpruefung. Kein Downstream-Vergleich. Die Ausschnitte je Lauf sind um dessen Defekt zentriert; sie zeigen nicht zwingend dieselbe Bildstelle. Bei fuenf neuen Hybriden pro Ziel zeigt die Tafel jeweils Index0.','', '| Kennzahl | Vorher | Neu |','|---|---:|---:|']
for k in ['hybrids','placements','real_rois_used','synthetic_variants_used','capture_groups_used','median_mask_pixels','median_inside_change','hybrids_changed_outside_mask_gt1','median_normalized_source_distance']:lines.append(f"| {k} | {summaries['vorher'][k]} | {summaries['neu'][k]} |")
lines+=['','## Quellenabdeckung','',f"Vorher fehlend: {summaries['vorher']['missing_capture_groups']}",f"Neu fehlend: {summaries['neu']['missing_capture_groups']}",'','## Einordnung','']
a,b=summaries['vorher'],summaries['neu'];lines.append('Die Aufnahmegruppenabdeckung ist verbessert.' if b['capture_groups_used']>a['capture_groups_used'] else 'Die Anzahl vertretener Aufnahmegruppen ist nicht gestiegen.');lines.append('Eine groessere Hybridanzahl allein belegt keine bessere Platzierungsqualitaet. Die veraenderten Matchinggewichte und LBM-Harmonisierung wirken gleichzeitig; der jeweilige Einfluss ist nicht isoliert. Geringere Distanz misst Positionsnaehe, nicht Defektplausibilitaet. Sichtpruefung der gepaarten Bilder und anschliessendes Downstream-Experiment sind fuer eine abschliessende Bewertung erforderlich.');lines+=['','Keine bestehenden Studien wurden durch diese Auswertung veraendert.'];(out/'README.md').write_text('\n'.join(lines)+'\n')
