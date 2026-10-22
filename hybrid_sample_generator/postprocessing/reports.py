"""Portable CSV reports and optional 2D object/anomaly previews."""
import csv
import numpy as np
from PIL import Image

def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['hybrid_id','accepted','reasons','measurements'])
        writer.writeheader();writer.writerows(rows)

def preview(path,image,mask,foreground):
    if image.ndim!=3:return
    rgb=np.moveaxis(np.repeat(image,3,axis=0) if image.shape[0]==1 else image,0,-1).clip(0,255).astype('uint8')
    if foreground is not None:rgb[foreground]=(rgb[foreground]*.75+np.array([0,100,255])*.25).astype('uint8')
    anomaly=(mask>0).any(0);rgb[anomaly]=(rgb[anomaly]*.3+np.array([255,0,0])*.7).astype('uint8')
    im=Image.fromarray(rgb);im.thumbnail((1200,1200));im.save(path)
