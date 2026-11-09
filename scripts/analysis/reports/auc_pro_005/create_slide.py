
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
from pptx.enum.text import PP_ALIGN
r=Presentation();r.slide_width=Inches(13.333);r.slide_height=Inches(7.5);s=r.slides.add_slide(r.slide_layouts[6])
C={'bg':'F3F6FA','ink':'14263D','muted':'5A6C80','teal':'087F8C','light':'DDF1F0','green':'23A68A','miss':'FFD980','red':'DD6673','normal':'E3E9F0','grid':'DCE4EC','white':'FFFFFF'}
s.background.fill.solid();s.background.fill.fore_color.rgb=RGBColor.from_string(C['bg'])
def box(x,y,w,h,color,line=None,rounded=False):
 a=s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE,Inches(x),Inches(y),Inches(w),Inches(h));a.fill.solid();a.fill.fore_color.rgb=RGBColor.from_string(color);a.line.fill.background()
 if line:a.line.color.rgb=RGBColor.from_string(line)
 if rounded:a.adjustments[0]=.12
 return a
def txt(x,y,w,h,t,size=14,color=None,bold=False,align=PP_ALIGN.LEFT):
 a=s.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h));tf=a.text_frame;tf.word_wrap=True;tf.margin_left=tf.margin_right=Inches(.015);tf.margin_top=Inches(.02);tf.margin_bottom=0
 for i,line in enumerate(t.split('\n')):
  p=tf.paragraphs[0] if i==0 else tf.add_paragraph();p.text=line;p.font.name='Aptos';p.font.size=Pt(size);p.font.bold=bold;p.font.color.rgb=RGBColor.from_string(color or C['ink']);p.alignment=align
 return a
def line(x1,y1,x2,y2,color,width=1):
 a=s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,Inches(x1),Inches(y1),Inches(x2),Inches(y2));a.line.color.rgb=RGBColor.from_string(color);a.line.width=Pt(width);return a
def dot(x,y,rad,color):return box(x-rad,y-rad,2*rad,2*rad,color,rounded=True)
box(0,0,13.333,.12,C['teal']);txt(.5,.35,12,.55,'AUC-PRO₀.₀₅: Defekte finden bei wenigen Fehlalarmen',29,bold=True);txt(.53,1.02,12,.4,'Misst die mittlere Abdeckung echter Defektregionen über alle Schwellenwerte bis 5 % Pixel-Fehlalarmrate.',14,C['muted'])
box(.5,1.7,5.55,4.69,C['white'],rounded=True);box(6.3,1.7,6.53,4.69,C['white'],rounded=True)
txt(.75,1.92,5.0,.37,'1   Jede Defektregion zählt gleich viel',18,bold=True)
# Exactly 100 pixels: two true regions (4 and16 pixels),80healthy.
small={(y,x) for y in (1,2) for x in (1,2)};large={(y,x) for y in range(5,9) for x in range(5,9)};detected={(1,1),(1,2),(2,1)}|{(y,x) for y in (5,6) for x in range(5,9)};fp={(0,8),(3,4),(8,1),(9,9)}
for y in range(10):
 for x in range(10):
  p=(y,x);col=C['green'] if p in detected else C['miss'] if p in small|large else C['red'] if p in fp else C['normal'];box(.95+x*.28,2.45+y*.28,.255,.255,col)
for xx,yy,ww,hh in [(1.215,2.715,.575,.575),(2.335,3.835,1.135,1.135)]:
 a=box(xx,yy,ww,hh,C['white'],C['ink']);a.fill.background();a.line.width=Pt(1.3)
for j,(col,title) in enumerate([(C['green'],'Defekt erkannt'),(C['miss'],'Defekt übersehen'),(C['red'],'Fehlalarm'),(C['normal'],'Gesund, korrekt')]):
 box(4.05,2.62+j*.51,.16,.16,col);txt(4.31,2.55+j*.51,1.48,.4,title,11,C['muted'])
txt(.95,5.42,4.85,.32,'Region A: 3/4 = 75 %   ·   Region B: 8/16 = 50 %',12,C['ink']);txt(.95,5.8,4.8,.35,'PRO = (75 % + 50 %) / 2 = 62,5 %',17,C['teal'],True)
# Partial PRO curve, an illustrative toy curve ending at the depicted mask.
txt(6.55,1.92,6,.37,'2   Schwellen variieren und Fläche messen',18,bold=True)
x0,y0,w,h=7.22,5.32,4.9,2.66
xs=[0,.005,.01,.02,.03,.04,.05];ys=[0,.25,.4,.5,.56,.6,.625];pts=[(x0+w*x/.05,y0-h*y) for x,y in zip(xs,ys)]
poly=[(x0,y0)]+pts[1:]+[(x0+w,y0)];ff=s.shapes.build_freeform(poly[0][0]*1000,poly[0][1]*1000,scale=Inches(1)/1000);ff.add_line_segments([(x*1000,y*1000) for x,y in poly[1:]],close=True);shape=ff.convert_to_shape();shape.fill.solid();shape.fill.fore_color.rgb=RGBColor.from_string(C['light']);shape.line.fill.background()
for t in [0,.25,.5,.75,1]:
 yy=y0-h*t;line(x0,yy,x0+w,yy,C['grid'],.6);txt(6.52,yy-.12,.56,.25,f'{t*100:g} %',10,C['muted'],align=PP_ALIGN.RIGHT)
for k in range(6):
 xx=x0+w*k/5;line(xx,y0,xx,y0+.06,C['muted'],.9);txt(xx-.25,y0+.12,.5,.28,f'{k} %',10,C['muted'],align=PP_ALIGN.CENTER)
line(x0,y0,x0,y0-h,C['muted'],1.3);line(x0,y0,x0+w,y0,C['muted'],1.3)
for (x1,y1),(x2,y2) in zip(pts,pts[1:]):line(x1,y1,x2,y2,C['teal'],2.7)
end=pts[-1];dot(*end,.055,C['teal']);txt(10.3,2.74,2.2,.62,'Beispiel bei 5 % FPR:\n62,5 % Regionsabdeckung',11,C['teal'],True);line(11.8,3.39,end[0],end[1]-.08,C['teal'],.9)
txt(7.23,2.28,4.5,.3,'PRO = mittlere Regionsabdeckung',11,C['muted']);txt(7.35,5.83,4.7,.32,'FPR = Fehlalarmpixel / alle gesunden Pixel',12,C['muted'],align=PP_ALIGN.CENTER)
area=sum((xs[i+1]-xs[i])*(ys[i+1]+ys[i])/2 for i in range(len(xs)-1))/.05
box(8.07,4.35,3.43,.61,C['white'],rounded=True);txt(8.15,4.48,3.27,.35,f'Normierte Fläche: {area*100:.2f} %'.replace('.',','),17,C['teal'],True,PP_ALIGN.CENTER)
box(.5,6.57,12.33,.58,C['ink'],rounded=True);txt(.72,6.69,7.6,.34,'AUC-PRO₀.₀₅ = Fläche bis 5 % FPR ÷ 0,05',15,C['white'],True);txt(8.55,6.69,4.0,.34,'0–100 %  ·  höher ist besser',15,C['white'],True,PP_ALIGN.RIGHT)
txt(.54,7.25,12,.17,'Illustratives Rechenbeispiel: 20 Defektpixel, 80 gesunde Pixel; 4 Fehlalarmpixel / 80 = 5 % FPR. Kurve schematisch.',9,C['muted'])
notes=s.notes_slide.notes_text_frame
notes.text='''PRO: Für jede zusammenhängende echte Defektregion wird der Anteil erkannter Pixel berechnet. Über alle Regionen gleichgewichtet mitteln; große Regionen zählen nicht stärker als kleine. FPR: falsch positive Pixel geteilt durch alle gesunden Pixel, gepoolt über die Auswertung. Schwellenwert variieren, PRO gegen FPR auftragen, nur von FPR0 bis0.05 integrieren und durch0.05 normieren. Beispiel links: Region A3/4, Region B8/16 -> PRO0.625;4von80gesundenPixeln falschpositiv -> FPR0.05. Die illustrative Kurve hat normierte Fläche0.4795,also47.95%. Der Score ist die Fläche über den FPR-Bereich und nicht allein der PRO-Wert bei5%. Unsere Implementierung nutzt4096Histogrammbins,8-Nachbarschaft für Regionen und lineare Interpolation am Integrationslimit. Quelle: examples/mvtec_ad2/downstream/evaluation.py, RegionOverlapMetric. Alle Diagramme editierbare PowerPoint-Formen.'''
out=report_dir('auc_pro_005')/'auc_pro_005_visuell_erklaert.pptx';r.save(out);print(out);print('Area',area,'grid',len(small),len(large),len(fp))
