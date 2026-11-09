"""Rebuild publication assets from the frozen configuration evidence."""

# Keep explicit frozen PYTHONPATH entries ahead of the live checkout.
import sys as _analysis_sys
from pathlib import Path as _AnalysisPath
if __package__ in (None, ""):
    _analysis_sys.path.append(str(_AnalysisPath(__file__).resolve().parents[4]))
from scripts.analysis.paths import report_dir

from pathlib import Path
import csv
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT=report_dir('paper_hybrid_generator_2026-11-05')
rows=json.loads((ROOT/'configuration_evidence.json').read_text())
names={'can':'Can','fabric':'Fabric','fruit_jelly':'Fruit jelly','rice':'Rice','sheet_metal':'Sheet metal','vial':'Vial','wallplugs':'Wall plugs','walnuts':'Walnuts'}
records=[]
for row in rows:
 c=row['configuration'];p=row['best_attributes']['params'];m=c['matching'];g=c['generation']
 records.append(dict(category=row['category'],channels=c['extraction']['anomaly_size'][0],patch=c['extraction']['anomaly_size'][-1],
  levels=p['n_levels'],blocks=p['n_res_blocks'],spade_blocks=p['n_spade_blocks'],z_channels=p['z_channels'],latent_dim=p['bottleneck_dim'],
  dropout=p['dropout'],skip_alpha=p['skip_alpha'],skip_dropout=p['skip_dropout_p'],trial=row['best_trial'],best_epoch=row['best_attributes']['best_epoch'],
  variants=g['variants_per_real_anomaly'],variation=g['variation_strength'],nominal_defects=m['anomalies_per_hybrid'],deviation=m['max_anomalies_per_hybrid_deviation'],
  hybrids_per_control=m['hybrids_per_original'],intensity_weight=m['intensity_weight'],gradient_weight=m['gradient_weight'],distance_weight=m['distance_weight'],
  harmonizer=c['fusion']['parameters']['image_harmonization'] or 'none',selected_hybrids=row['selected_hybrids']))
with (ROOT/'category_settings.csv').open('w') as stream:
 writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
lines=[r'\begin{table*}[t]',r'\centering',r'\caption{Frozen V3 hybrid generator settings. $C$: image channels; $P$: generative patch side; $L/B$: levels/ConvNeXt blocks per level; $Z$: bottleneck feature channels; $d$: latent-vector dimension; $V$: variants per extracted donor; $s$: posterior variation scale; $K\pm\Delta$: requested defects per hybrid; $H$: requested hybrids per healthy control; $N$: hybrids used downstream. Placement constraints and filtering can reduce realized counts. LBM harmonization is enabled only for can and rice.}',r'\label{tab:hybrid-settings}',r'\footnotesize',r'\setlength{\tabcolsep}{4pt}',r'\begin{tabular}{@{}lrrrrrrrrrrr@{}}',r'\toprule',r'Category & $C$ & $P$ & $L/B$ & $Z$ & $d$ & $V$ & $s$ & $K\pm\Delta$ & $H$ & $N$ & Best epoch \\',r'\midrule']
for r in records:
 lines.append(f"{names[r['category']]} & {r['channels']} & {r['patch']} & {r['levels']}/{r['blocks']} & {r['z_channels']} & {r['latent_dim']} & {r['variants']} & {r['variation']:g} & ${r['nominal_defects']}\\pm{r['deviation']}$ & {r['hybrids_per_control']} & {r['selected_hybrids']} & {r['best_epoch']} "+r'\\')
lines += [r'\bottomrule',r'\end{tabular}',r'\end{table*}']
(ROOT/'settings_table.tex').write_text('\n'.join(lines)+'\n')
plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'svg.fonttype':'none'})
fig,ax=plt.subplots(figsize=(14.5,4.1));ax.set(xlim=(0,14.5),ylim=(0,4.1));ax.axis('off')
def box(x,y,w,h,title,body,color='#eaf1f8'):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.045,rounding_size=0.09',linewidth=1.1,edgecolor='#3c536c',facecolor=color))
 ax.text(x+w/2,y+h-.22,title,ha='center',va='top',fontsize=15,fontweight='bold',color='#162c41')
 ax.text(x+w/2,y+h/2-.15,body,ha='center',va='center',fontsize=13.5,linespacing=1.45,color='#243b50')
def arrow(a,b):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=15,color='#395773',linewidth=1.3))
xs=[.12,3.02,5.92,8.82,11.72];w=2.60;y=1.50;h=1.32
box(xs[0],y,w,h,'Training donors','Real defect images\nand binary masks')
box(xs[1],y,w,h,'Extract + normalize','Connected components\nScale / intensity metadata')
box(xs[2],y,w,h,'Conditional VAE','ConvNeXt encoder\nLatent sampling + SPADE')
box(xs[3],y,w,h,'Match + fuse','Donor-context matching\nScale restoration + blending')
box(xs[4],y,w,h,'Paired output','Hybrid + healthy control\nMask + full provenance')
for a,b in zip(xs,xs[1:]):arrow((a+w,y+h/2),(b-.04,y+h/2))
box(6.02,3.10,2.40,.78,'Target masks','Zoom / stretch / rotation',color='#eaf5ef')
arrow((7.22,3.10),(7.22,2.84))
box(8.92,.20,2.40,.87,'Healthy controls','Training partition only',color='#f8f0e5')
arrow((10.12,1.07),(10.12,1.48))
ax.text(.15,.51,'Fitting: training defects only; internal patch holdout.\nDownstream validation: separate acquisition groups.',fontsize=13,color='#3e5163',va='center')
ax.text(11.78,.70,'Can / rice: additional\npretrained LBM harmonization',fontsize=12,color='#3e5163',va='center')
arrow((11.9,.99),(11.4,1.50))
fig.subplots_adjust(left=.01,right=.99,bottom=.02,top=.98)
for suffix in ['pdf','svg','png']:fig.savefig(ROOT/'figures'/f'pipeline.{suffix}',dpi=180,facecolor='white')
plt.close(fig)
assert sum(r['selected_hybrids'] for r in records)==2599
print('Generated PDF/SVG/PNG diagram, LaTeX table, and CSV for all eight categories.')
