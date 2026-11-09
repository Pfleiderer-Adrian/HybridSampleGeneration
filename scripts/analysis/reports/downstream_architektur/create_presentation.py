
# Keep explicit frozen PYTHONPATH entries ahead of the live checkout.
import sys as _analysis_sys
from pathlib import Path as _AnalysisPath
if __package__ in (None, ""):
    _analysis_sys.path.append(str(_AnalysisPath(__file__).resolve().parents[4]))
from scripts.analysis.paths import report_dir

from pathlib import Path
from pptx import Presentation
from pptx.util import Inches,Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE,MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN,MSO_ANCHOR
from pptx.oxml.xmlchemy import OxmlElement
R=Presentation();R.slide_width=Inches(13.333);R.slide_height=Inches(7.5)
NAVY='14263D';TEAL='087F8C';PURPLE='7256A8';GOLD='BC7B20';INK='26374B';MUTED='617186';BG='F4F7FB';WHITE='FFFFFF';LINE='CCD6E3'
def box(s,x,y,w,h,fill=WHITE,line=None,r=True):
 a=s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if r else MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h));a.fill.solid();a.fill.fore_color.rgb=RGBColor.from_string(fill);a.line.fill.background() if line is None else None
 if line:a.line.color.rgb=RGBColor.from_string(line)
 if r:
  try:a.adjustments[0]=.14
  except Exception:pass
 return a
def text(s,x,y,w,h,t,size=14,color=INK,bold=False,align=PP_ALIGN.LEFT):
 a=s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h));tf=a.text_frame;tf.word_wrap=True;tf.margin_left=tf.margin_right=Inches(.02);tf.margin_top=Inches(.02)
 for i,l in enumerate(t.split('\n')):
  p=tf.paragraphs[0] if i==0 else tf.add_paragraph();p.text=l;p.alignment=align;p.font.name='Aptos';p.font.size=Pt(size);p.font.bold=bold;p.font.color.rgb=RGBColor.from_string(color)
 return a
def arrow(s,x1,y1,x2,y2,color=MUTED,dash=False):
 a=s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2));a.line.color.rgb=RGBColor.from_string(color);a.line.width=Pt(1.7)
 ln=a._element.spPr.get_or_add_ln();head=OxmlElement('a:tailEnd');head.set('type','triangle');ln.append(head)
 if dash:
  from pptx.enum.dml import MSO_LINE_DASH_STYLE
  a.line.dash_style=MSO_LINE_DASH_STYLE.DASH
 return a
def node(s,x,y,w,h,title,sub,fill=WHITE,color=INK):
 box(s,x,y,w,h,fill);text(s,x+.12,y+.1,w-.24,.33,title,15,color,True);text(s,x+.12,y+.48,w-.24,h-.5,sub,11,color)
def slide(n,title,sub):
 s=R.slides.add_slide(R.slide_layouts[6]);s.background.fill.solid();s.background.fill.fore_color.rgb=RGBColor.from_string(BG)
 box(s,0,0,13.333,.12,TEAL,r=False);text(s,.5,.35,12,.55,title,27,NAVY,True);text(s,.52,1.02,12,.45,sub,13,MUTED);text(s,.52,7.12,11,.25,'Hybrid Sample Generator  •  DINOv3 Paired / einfacher CNN-Decoder',10,MUTED);text(s,12.1,7.08,.7,.3,f'{n:02}',12,MUTED,align=PP_ALIGN.RIGHT);return s
s=slide(1,'Paarweises Training: Defekt erkennen, Hintergrund stabil halten','Zwei Bilder durch dasselbe Modell · gemeinsame Gewichte · der DINO-Encoder bleibt eingefroren')
for y,title,sub,col in [(1.8,'Synthetisches Defektbild','Hybrid / DRAEM / CutMix',TEAL),(3.6,'Gesundes Partnerbild','Original vor dem Einfügen',PURPLE)]:
 node(s,.55,y,2.45,1.18,title,sub,col,WHITE);node(s,3.55,y,5.0,1.18,'Geteiltes Segmentierungsmodell','DINOv3-Features + RGB-Detailpfad → CNN-Decoder',WHITE)
 arrow(s,3.02,y+.58,3.51,y+.58,col);arrow(s,8.6,y+.58,9.06,y+.58,col)
 node(s,9.1,y,3.65,1.18,'Defektvorhersage' if col==TEAL else 'Vorhersage für gesundes Bild','Wahrscheinlichkeit pro Pixel' if col==TEAL else 'Ziel: überall 0 = gesund',col,WHITE)
# consistency bridge
arrow(s,10.95,3.0,10.95,3.55,PURPLE);box(s,6.85,3.1,3.66,.4,BG);text(s,6.9,3.12,3.6,.34,'Konsistenz nur außerhalb der Defektmaske',11,PURPLE,True)
for x,title,sub,col in [(.55,'01  Focal + Dice','Defektvorhersage ↔ synthetische Defektmaske',TEAL),(4.72,'02  Partner-Loss','Focal: gesundes Partnerbild ↔ leere Maske',PURPLE),(8.89,'03  Hintergrundkonsistenz','Quadratische Differenz der Vorhersagen außerhalb der Maske',GOLD)]:
 node(s,x,5.15,3.88,1.05,title,sub,WHITE,col)
text(s,.62,6.47,12,.45,'Gesamtverlust = Focal + Dice + Partner-Loss + Konsistenz   ·   Gewichte aktuell jeweils 1,0',15,NAVY,True)
text(s,.62,6.87,11.9,.22,'Echte Defekte: Focal + Dice ohne Partner. Gesunde Einzelbeispiele bei pair_healthy_samples=False: nur Focal.',10,MUTED)
s=slide(2,'Der RGB-Detailpfad ergänzt die DINO-Merkmale','Aktueller V3-Decoder: cnn, Breite 64 · der Detailpfad erhält feine lokale Strukturen unterhalb eines ViT-Patches')
node(s,.55,3.1,1.65,1.05,'RGB-Patch','3 × 256 × 256',NAVY,WHITE)
node(s,2.75,1.8,2.65,1.23,'DINOv3 Small/16','Eingefroren · letzte Schicht\n16 × 16 räumliche Features',WHITE)
node(s,5.85,1.8,2.65,1.23,'Projektion + Upsampling','1×1 Conv → 64 Kanäle\nBilinear → 256 × 256',WHITE)
node(s,2.75,4.1,3.38,1.23,'RGB-Detailpfad','Trainierbar: 3×3 Conv (3 → 64)\nGELU · Auflösung 256 × 256',TEAL,WHITE)
arrow(s,2.23,3.6,2.48,3.6);arrow(s,2.48,3.6,2.48,2.42);arrow(s,2.48,2.42,2.71,2.42)
arrow(s,2.48,3.6,2.48,4.72,TEAL);arrow(s,2.48,4.72,2.71,4.72,TEAL);arrow(s,5.44,2.42,5.81,2.42)
node(s,8.95,3.1,1.83,1.05,'Concat','64 + 64 = 128\nFeaturekanäle',PURPLE,WHITE)
arrow(s,8.54,2.42,9.5,2.42);arrow(s,9.5,2.42,9.5,3.06)
arrow(s,6.18,4.72,9.5,4.72,TEAL);arrow(s,9.5,4.72,9.5,4.19,TEAL)
node(s,11.2,3.1,1.58,1.05,'CNN-Kopf','3×3: 128 → 64\nGELU · 1×1: 64 → 2',NAVY,WHITE);arrow(s,10.82,3.62,11.16,3.62)
node(s,9.0,5.43,3.78,.95,'Softmax → Anomaliekarte','Klasse „Defekt“ · 1 × 256 × 256',WHITE)
arrow(s,12.0,4.2,12.0,5.39)
box(s,.55,5.9,7.75,.8,WHITE);text(s,.72,6.03,7.4,.53,'Ein Bild genügt bei der Inferenz. Der gesunde Partner wird nur für das Training benötigt.',15,NAVY,True)
text(s,2.79,5.49,5.7,.27,'Beide Pfade werden bei jedem Bild gemeinsam verwendet.',11,MUTED)
p=report_dir('downstream_architektur')/'dinov3_paired_training_rgb_detailpfad.pptx';R.save(p);print(p)
