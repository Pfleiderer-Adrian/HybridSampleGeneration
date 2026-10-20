"""Read-only V2/V3 Can placement comparison, matched targets and white background."""
import pathlib,sqlite3,json,collections,re,csv
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import binary_propagation,label,find_objects
ROOT=pathlib.Path('/mnt/results/mvtec2/experiments');OUT=pathlib.Path(__file__).parent

def mask2(a):
 a=np.asarray(a)>0
 return a.any(0) if a.ndim==3 else a

def rgb(a):return np.clip(np.moveaxis(a,0,-1),0,255).astype('uint8')
def load(study,p):return np.load(study/p,mmap_mode='r')
def cropbox(m):
 y,x=np.where(m);cy=(int(y.min())+int(y.max()))//2;cx=(int(x.min())+int(x.max()))//2
 # Bound the crop around largest mask component to avoid shrinking distant defects.
 lab,n=label(m);counts=np.bincount(lab.ravel());counts[0]=0;largest=int(counts.argmax());slices=find_objects(lab)[largest-1];ys,xs=slices;cy=(ys.start+ys.stop)//2;cx=(xs.start+xs.stop)//2
 h=max(256,ys.stop-ys.start+80);w=max(256,xs.stop-xs.start+80)
 return max(0,cx-w//2),max(0,cy-h//2),min(m.shape[1],cx+(w+1)//2),min(m.shape[0],cy+(h+1)//2)
def panel(image,size=(320,265)):
 image.thumbnail((size[0]-8,size[1]-8));p=Image.new('RGB',size,'#222222');p.paste(image,((size[0]-image.width)//2,(size[1]-image.height)//2));return p
studies={};summaries={};metrics={}
for version in ['comparison_v2','comparison_v3']:
 study=ROOT/version/'can/hybrid_generation';db=sqlite3.connect('file:'+str(study/'artifacts.sqlite')+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
 tables={t:{row['id']:dict(row) for row in db.execute('select * from '+t)} for t in ['original_samples','hybrid_samples','placements','real_anomalies','synthetic_anomalies']};h=[x for x in tables['hybrid_samples'].values() if x['status']=='generated' and not tables['original_samples'][x['original_sample_id']]['has_anomaly']];ps=collections.defaultdict(list)
 for p in tables['placements'].values():ps[p['hybrid_sample_id']].append(p)
 donor_counts=collections.Counter();capture_counts=collections.Counter();synthetics=collections.Counter();rows={};all_dists=[]
 for i,hybrid in enumerate(h):
  original=tables['original_samples'][hybrid['original_sample_id']];a=load(study,original['image_path']);m=mask2(load(study,hybrid['segmentation_path']));white=(np.min(a,axis=0)>=245);seed=np.zeros_like(white);seed[0]=white[0];seed[-1]=white[-1];seed[:,0]=white[:,0];seed[:,-1]=white[:,-1];bg=binary_propagation(seed,mask=white);pcount=int(m.sum())
  used=[];dists=[]
  for placement in ps[hybrid['id']]:
   s=tables['synthetic_anomalies'][placement['synthetic_anomaly_id']];real=tables['real_anomalies'][s['real_anomaly_id']];donor=tables['original_samples'][real['original_sample_id']];donor_counts[donor['source_name']]+=1;capture_counts[re.sub(r'_(regular|overexposed|underexposed|shift_\d+)\.png$','',donor['source_name'])]+=1;synthetics[s['id']]+=1;used.append(donor['source_name']);dist=float(np.hypot(placement['position_y']-real['position_y'],placement['position_x']-real['position_x']));dists.append(dist);all_dists.append(dist)
  row={'hybrid_id':hybrid['id'],'target':original['source_name'],'placements':len(ps[hybrid['id']]),'mask_pixels':pcount,'white_background_fraction':float((m&bg).sum()/max(1,pcount)),'minimum_score':min(p['score'] for p in ps[hybrid['id']]),'mean_source_distance':float(np.mean(dists)),'donors':'|'.join(used)};rows[original['source_name']]=row
  if i%50==0:print(version,i,'/',len(h),flush=True)
 studies[version]=(study,tables,ps,rows);metrics[version]=rows
 summaries[version]={'hybrids':len(h),'placements':sum(r['placements'] for r in rows.values()),'donor_images_used':len(donor_counts),'donor_capture_groups_used':len(capture_counts),'capture_usage':capture_counts,'top_synthetic_usage':synthetics.most_common(5),'unique_synthetics_used':len(synthetics),'median_normalized_source_distance':float(np.median(all_dists)),'hybrids_any_white_background_overlap':sum(r['white_background_fraction']>0 for r in rows.values()),'hybrids_majority_white_background':sum(r['white_background_fraction']>.5 for r in rows.values()),'hybrids_over_10pct_white_background':sum(r['white_background_fraction']>.1 for r in rows.values()),'study':str(study),'configuration':json.loads((study/'configuration.json').read_text())}
 with (OUT/(version+'_metrics.csv')).open('w') as f:w=csv.DictWriter(f,fieldnames=list(next(iter(rows.values()))));w.writeheader();w.writerows(rows.values())
common=sorted(set(metrics['comparison_v2'])&set(metrics['comparison_v3']));rng=np.random.default_rng(42);random_names=[common[i] for i in rng.choice(len(common),24,replace=False)];flagged=sorted(common,key=lambda n:max(metrics[v][n]['white_background_fraction'] for v in studies),reverse=True)[:8]
comparison={'common_targets':len(common),'random_sample_targets':random_names,'flagged_targets':flagged,'common_target_statistics':{v:{'any_white':sum(metrics[v][n]['white_background_fraction']>0 for n in common),'majority_white':sum(metrics[v][n]['white_background_fraction']>.5 for n in common),'over_10pct_white':sum(metrics[v][n]['white_background_fraction']>.1 for n in common)} for v in studies}}

def draw_sheet(names,destination):
 sheet=Image.new('RGB',(1280,35+len(names)*300),'white');draw=ImageDraw.Draw(sheet)
 for i,t in enumerate(['V2 target + red location','V3 target + blue location','V2 hybrid closeup','V3 hybrid closeup']):draw.text((i*320+5,8),t,fill='black')
 for row,name in enumerate(names):
  panels=[]
  for version,color in [('comparison_v2',(255,0,0)),('comparison_v3',(0,100,255))]:
   study,tables,ps,records=studies[version];rec=records[name];hybrid=tables['hybrid_samples'][rec['hybrid_id']];orig=tables['original_samples'][hybrid['original_sample_id']];original=rgb(load(study,orig['image_path']));actual=rgb(load(study,hybrid['image_path']));m=mask2(load(study,hybrid['segmentation_path']));overlay=original.copy();overlay[m]=(overlay[m]*.3+np.array(color)*.7).astype('uint8');im=Image.fromarray(overlay);d=ImageDraw.Draw(im)
   for p in ps[hybrid['id']]:
    y=p['position_y']*m.shape[0];x=p['position_x']*m.shape[1];d.ellipse((x-30,y-30,x+30,y+30),outline=color,width=5)
   panels.append(panel(im));panels.append(panel(Image.fromarray(actual).crop(cropbox(m))))
  for j,index in enumerate([0,2,1,3]):sheet.paste(panels[index],(j*320,35+row*300))
  v2=metrics['comparison_v2'][name];v3=metrics['comparison_v3'][name];draw.text((5,302+row*300),f"{name}   white overlap V2={v2['white_background_fraction']:.0%}, V3={v3['white_background_fraction']:.0%}",fill='black')
 sheet.save(destination,quality=92)
for i in range(6):draw_sheet(random_names[i*4:i*4+4],OUT/f'paired_random_{i+1}.jpg')
for i in range(2):draw_sheet(flagged[i*4:i*4+4],OUT/f'paired_white_background_{i+1}.jpg')
(OUT/'summary.json').write_text(json.dumps({'studies':summaries,'comparison':comparison,'white_background_definition':'original pixels min(R,G,B)>=245 connected to image boundary; heuristic, not object ground truth; masks measured on actual materialized foreground','distance_definition':'Euclidean distance of normalized ROI matching center to original real-anomaly position; 0..sqrt(2); not the weighted score component'},indent=2)+'\n')
print('COMPLETE',json.dumps({'studies':{k:{a:b for a,b in v.items() if a not in ['configuration','capture_usage','top_synthetic_usage']} for k,v in summaries.items()},'comparison':comparison}),flush=True)
