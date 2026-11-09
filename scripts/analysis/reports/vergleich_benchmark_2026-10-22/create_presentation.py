"""Export completed benchmark validation AUC-PRO results as editable PPTX tables."""

# Keep explicit frozen PYTHONPATH entries ahead of the live checkout.
import sys as _analysis_sys
from pathlib import Path as _AnalysisPath
if __package__ in (None, ""):
    _analysis_sys.path.append(str(_AnalysisPath(__file__).resolve().parents[4]))
from scripts.analysis.paths import report_dir

import json,csv
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches,Pt
from pptx.dml.color import RGBColor
ROOT=Path('/mnt/results/mvtec2/experiments/vergleich_benchmark');OUT=report_dir('vergleich_benchmark_2026-10-22')
CATS=['can','fabric','fruit_jelly','rice','sheet_metal','vial','wallplugs','walnuts']
NAMES=['Can','Fabric','Fruit Jelly','Rice','Sheet Metal','Vial','Wallplugs','Walnuts']
VARIANTS=['01_real_only','02_real_draem','03_real_cutmix','04_real_hybrid','05_real_hybrid_draem','06_real_hybrid_draem_cutmix']
records=[]
def record(path,kind='Original'):
 s=json.loads((path/'status.json').read_text());assert s['stage']=='complete',path
 score=s['metrics']['validation']['pixel_aupro_0.05'];cfg=json.loads((path/'downstream_configuration.json').read_text())
 r={'category':s['category'],'variant':s['variant'],'run_type':kind,'score':score,'selected_epoch':s['selected_epoch'],'source_quotas':cfg['data']['source_quotas'],'source_quota_rotation':cfg['data'].get('source_quota_rotation'),'run':s['downstream_run'],'status_path':str(path/'status.json')};records.append(r);return r
main={c:[record(ROOT/v/c) for v in VARIANTS] for c in CATS}
reruns={}
for c in ['can','rice']:
 info=json.loads((ROOT/f'latest_{c}_hybrid_rerun.json').read_text());reruns[c]=[record(ROOT/k,'Mehr Hybride') for k in info['benchmark_jobs']]
(OUT/'results.json').write_text(json.dumps(records,indent=2)+'\n')
with (OUT/'results.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=['category','variant','run_type','score_percent','selected_epoch','source_quotas','run']);w.writeheader()
 for r in records:w.writerow({k:r[k] for k in ['category','variant','run_type','selected_epoch','run']}|{'score_percent':r['score']*100,'source_quotas':json.dumps(r['source_quotas'])})
prs=Presentation();prs.slide_width=Inches(16);prs.slide_height=Inches(9)
WHITE='FFFFFF';INK='192D40';GRAY='E8EDF2';GREEN='DDF1E2';RED='F7DEDE';GTEXT='176139';RTEXT='A32929';BLUE='233F58'
def box(slide,x,y,w,h,text,size=18,color=INK,bold=False):
 tf=slide.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h)).text_frame;tf.word_wrap=True
 for i,line in enumerate(text.split('\n')):
  p=tf.paragraphs[0] if i==0 else tf.add_paragraph();p.text=line;p.font.name='Aptos';p.font.size=Pt(size);p.font.bold=bold;p.font.color.rgb=RGBColor.from_string(color)
 return tf

def slide(title,subtitle):
 s=prs.slides.add_slide(prs.slide_layouts[6]);box(s,.55,.35,15,.6,title,29,bold=True);box(s,.55,1.05,15,.65,subtitle,15)
 return s

def table(s,headers,rows,widths,y=1.95,rowheight=.58,headerheight=1.2):
 t=s.shapes.add_table(len(rows)+1,len(headers),Inches(.55),Inches(y),Inches(sum(widths)),Inches(headerheight+len(rows)*rowheight)).table
 for i,w in enumerate(widths):t.columns[i].width=Inches(w)
 for ri,row in enumerate([headers]+rows):
  t.rows[ri].height=Inches(headerheight if ri==0 else rowheight)
  for ci,item in enumerate(row):
   cell=t.cell(ri,ci);cell.margin_left=cell.margin_right=Inches(.08);cell.margin_top=cell.margin_bottom=Inches(.07)
   if ri==0:text,bg,fg,bold=str(item),BLUE,WHITE,True
   elif isinstance(item,tuple):
    value,base=item;delta=value-base;neutral=abs(delta)<1e-12;text=f'{value*100:.2f} %'.replace('.',',');bg=GRAY if neutral else GREEN if delta>0 else RED;fg=INK if neutral else GTEXT if delta>0 else RTEXT;bold=False
   else:text,bg,fg,bold=str(item),GRAY,INK,True
   cell.fill.solid();cell.fill.fore_color.rgb=RGBColor.from_string(bg);cell.text=text
   from pptx.enum.text import MSO_ANCHOR,PP_ALIGN
   cell.vertical_anchor=MSO_ANCHOR.MIDDLE
   for p in cell.text_frame.paragraphs:
    p.alignment=PP_ALIGN.LEFT if ci==0 else PP_ALIGN.CENTER;p.font.name='Aptos';p.font.size=Pt(15 if ri==0 else 20);p.font.bold=bold;p.font.color.rgb=RGBColor.from_string(fg)
 return t
s=slide('Datenquellen im Vergleich – alle Kategorien','Validierungs-AUC-PRO₀.₀₅ · DINOv3 Small + einfacher CNN-Decoder · gruppierter 60/40-Split · Stand: 22.10.2026')
headers=['Kategorie','1 · Nur echt\n50 % gesund\n50 % echte Defekte','2 · Echt + DRAEM\n50 % / 10 % / 40 %','3 · Echt + CutMix\n50 % / 10 % / 40 %','4 · Echt + Hybrid\n50 % / 10 % / 40 %','5 · Hybrid + DRAEM\n50 % / 10 %\n20 % H + 20 % D','6 · Hybrid + DRAEM\n+ CutMix\n50 % / 10 % / je 13⅓ %']
rows=[[n]+[(r['score'],main[c][0]['score']) for r in main[c]] for c,n in zip(CATS,NAMES)]
table(s,headers,rows,[1.75]+[2.19]*6,rowheight=.57,headerheight=1.25)
box(s,.6,7.95,15,.65,'Farben je Kategorie gegenüber „Nur echt“: Grau = Referenz / gleich · Grün = höher · Rot = niedriger.\nMischung ab Variante 2: gesund / echte Defekte / synthetisch. H = Hybrid, D = DRAEM; Variante 6 rotiert 134/133/133 pro Epoche.',13)
box(s,.6,8.65,15,.25,'Alle Läufe abgeschlossen · beste Epoche nach Validierungs-AUC-PRO₀.₀₅ · keine privaten Testwerte · Originale Hybrid-Datensätze',11)
s=slide('Can & Rice – Wiederholungen mit mehr Hybridbildern','Gleiche Downstream-Konfiguration und gleicher Split · Generator wiederverwendet · Farben erneut gegenüber „Nur echt“')
headers=['Kategorie','Nur echt\n50 % gesund\n50 % echt','Echt + DRAEM\n50/10/40','4 · Hybrid\nvorher\n50/10/40','4 · Hybrid\nneu\n50/10/40','5 · H + D\nvorher\n50/10/20/20','5 · H + D\nneu\n50/10/20/20','6 · H + D + C\nvorher\n50/10/je 13⅓','6 · H + D + C\nneu\n50/10/je 13⅓']
rows=[]
for c,n in [('can','Can'),('rice','Rice')]:
 base=main[c][0]['score'];values=[main[c][0],main[c][1],main[c][3],reruns[c][0],main[c][4],reruns[c][1],main[c][5],reruns[c][2]];rows.append([n]+[(v['score'],base) for v in values])
table(s,headers,rows,[1.35]+[1.7]*8,y=2,rowheight=.9,headerheight=1.4)
box(s,.6,5.6,14.8,1.25,'Neue Hybriddaten: Can 197 → 636 Hybridbilder · Rice 234 → 468 Hybridbilder.\nMischungsverhältnisse unverändert: 50 % gesund + 10 % echte Defekte + insgesamt 40 % synthetische Beispiele.\nVariante 5: je 20 % Hybrid/DRAEM · Variante 6: je ca. 13⅓ % Hybrid/DRAEM/CutMix.',18)
box(s,.6,7.4,14.8,.8,'Grün bedeutet besser als „Nur echt“ – nicht automatisch besser als DRAEM oder als der frühere Hybrid-Lauf.\nH = Hybrid · D = DRAEM · C = CutMix. Angaben sind Anteile aller Trainingsbeispiele.',15)
box(s,.6,8.55,15,.3,'Quelldaten und Run-Pfade: results.json / results.csv neben dieser Präsentation · 54 abgeschlossene Läufe insgesamt',11)
path=OUT/'vergleich_benchmark_alle_ergebnisse_2026-10-22.pptx';prs.save(path)
# Verify editable tables, exact displayed data count, and per-cell color coding.
check=Presentation(path);assert len(check.slides)==2;count=0
for sl in check.slides:
 for shape in sl.shapes:
  if shape.has_table:
   for row in list(shape.table.rows)[1:]:
    for cell in list(row.cells)[1:]:
     assert '%' in cell.text;assert str(cell.fill.fore_color.rgb) in [GRAY,GREEN,RED];count+=1
assert count==64,count
assert len(records)==54
print(path);print('Verified 54 source results and 64 numeric table cells, two editable tables.')
