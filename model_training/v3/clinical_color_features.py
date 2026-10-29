"""Label-free chromaticity/texture descriptors; never a clinical segmentation mask."""
import cv2
import numpy as np


def clinical_color_features(rgb):
    rgb=np.asarray(rgb,dtype=np.float32)
    if rgb.ndim!=3 or rgb.shape[2]!=3 or min(rgb.shape[:2])<8 or not np.isfinite(rgb).all() or rgb.min()<-.00001 or rgb.max()>1.00001:
        raise ValueError('Expected finite HWC RGB in 0..1')
    rgb=np.clip(rgb,0,1);maximum=rgb.max(2);minimum=rgb.min(2)
    mask=((maximum-minimum)>.08)&(rgb[:,:,1]>.85*rgb[:,:,0])&(rgb[:,:,1]>.9*rgb[:,:,2])&(maximum>.04)
    fraction=float(mask.mean())
    if mask.sum()<32:mask=np.ones(rgb.shape[:2],bool)
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV);lab=cv2.cvtColor(rgb,cv2.COLOR_RGB2LAB)
    values=[]
    for channel,bins,limits in [(hsv[:,:,0],18,(0,360)),(hsv[:,:,1],8,(0,1)),(hsv[:,:,2],8,(0,1))]:
        h=np.histogram(channel[mask],bins=bins,range=limits)[0].astype(np.float32);values.extend(h/max(h.sum(),1))
    q=[.05,.25,.5,.75,.95]
    for channel in (lab[:,:,0]/100,lab[:,:,1]/128,lab[:,:,2]/128,(rgb[:,:,1]-rgb[:,:,0])/(rgb[:,:,1]+rgb[:,:,0]+.001),(rgb[:,:,1]-rgb[:,:,2])/(rgb[:,:,1]+rgb[:,:,2]+.001)):
        values.extend(np.quantile(channel[mask],q))
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY);lap=cv2.Laplacian(gray,cv2.CV_32F)
    values.extend(np.quantile(np.abs(lap[mask]),q));values.append(fraction)
    # Rotation-invariant uniform LBP: transitions <=2 use set-bit count; all
    # other patterns share one bin. The descriptor does not assign tissue labels.
    center=gray[1:-1,1:-1];neighbors=[gray[:-2,:-2],gray[:-2,1:-1],gray[:-2,2:],gray[1:-1,2:],gray[2:,2:],gray[2:,1:-1],gray[2:,:-2],gray[1:-1,:-2]]
    bits=np.stack([v>=center for v in neighbors]);transitions=(bits!=np.roll(bits,1,axis=0)).sum(0);code=np.where(transitions<=2,bits.sum(0),9)
    h=np.bincount(code[mask[1:-1,1:-1]],minlength=10).astype(np.float32);values.extend(h/max(h.sum(),1))
    result=np.asarray(values,dtype=np.float32);assert result.shape==(75,) and np.isfinite(result).all()
    return result
