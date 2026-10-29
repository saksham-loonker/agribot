"""Label-free ridge/colour contrasts; heuristic masks are not clinical segmentation."""
import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes,distance_transform_edt

DIMENSION=69

def structured_leaf_features(rgb):
    rgb=np.asarray(rgb,dtype=np.float32)
    if rgb.ndim!=3 or rgb.shape[2]!=3 or min(rgb.shape[:2])<16 or not np.isfinite(rgb).all() or rgb.min()<0 or rgb.max()>1:
        raise ValueError('Finite RGB image in0–1 units, at least16x16, required')
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV);sat=hsv[:,:,1];value=hsv[:,:,2]
    seed=((sat>.06)&(value>.05)).astype(np.uint8);seed=cv2.morphologyEx(seed,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)))
    count,labels,stats,_=cv2.connectedComponentsWithStats(seed,8)
    if count<2:return np.zeros(DIMENSION,np.float32)
    component=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]));mask=binary_fill_holes(labels==component)
    if int(mask.sum())<64:return np.zeros(DIMENSION,np.float32)
    distance=distance_transform_edt(np.pad(mask,1))[1:-1,1:-1];normalized=distance/max(float(distance.max()),1.);interior=mask&(distance>=max(2.,.02*min(mask.shape)))
    if int(interior.sum())<16:interior=mask
    total=rgb.sum(2).clip(1e-6);chroma=rgb/total[:,:,None];lum=(rgb*np.array([.2126,.7152,.0722],np.float32)).sum(2).astype(np.float64);lab=cv2.cvtColor(rgb,cv2.COLOR_RGB2LAB)
    channels=np.stack([lum,chroma[:,:,1],chroma[:,:,0],chroma[:,:,2],sat,lab[:,:,2]/128.],axis=-1).astype(np.float64)
    features=[]
    for k in range(6):features.extend([float(channels[:,:,k][mask].mean()),float(channels[:,:,k][mask].std())])
    bright=np.zeros(mask.shape,np.float64);dark=np.zeros_like(bright)
    for sigma in (1.,2.,4.):
        smooth=cv2.GaussianBlur(lum,(0,0),sigmaX=sigma,sigmaY=sigma);xx=cv2.Sobel(smooth,cv2.CV_64F,2,0,ksize=3)*sigma*sigma;yy=cv2.Sobel(smooth,cv2.CV_64F,0,2,ksize=3)*sigma*sigma;xy=cv2.Sobel(smooth,cv2.CV_64F,1,1,ksize=3)*sigma*sigma
        delta=np.sqrt((xx-yy)**2+4*xy*xy);one=(xx+yy-delta)/2;two=(xx+yy+delta)/2;small=np.where(abs(one)<=abs(two),one,two);large=np.where(abs(one)<=abs(two),two,one);strength=np.sqrt(small*small+large*large);scale=max(float(np.quantile(strength[interior],.9)),1e-8);response=np.exp(-2*np.square(small/(abs(large)+1e-12)))*(1-np.exp(-.5*np.square(strength/scale)))
        bright=np.maximum(bright,np.where(large<0,response,0));dark=np.maximum(dark,np.where(large>0,response,0))
    h,w=mask.shape
    for response in (bright,dark):
        response=np.round(response,6);local=response[interior];features.extend(np.quantile(local,[.5,.75,.9,.99]).tolist());cutoff=float(np.quantile(local,.85));ridge=interior&(response>=cutoff)&(response>1e-6);background=interior&~ridge;features.append(float(ridge.sum()/interior.sum()))
        for q in (None,.75):
            for k in range(6):
                if not ridge.any() or not background.any():features.append(0.)
                elif q is None:features.append(float(channels[:,:,k][ridge].mean()-channels[:,:,k][background].mean()))
                else:features.append(float(np.quantile(channels[:,:,k][ridge],q)-np.quantile(channels[:,:,k][background],q)))
        quadrants=[]
        for y0,y1 in ((0,h//2),(h//2,h)):
            for x0,x1 in ((0,w//2),(w//2,w)):
                m=interior[y0:y1,x0:x1];quadrants.append(float(response[y0:y1,x0:x1][m].mean()) if m.any() else 0.)
        features.extend(sorted(quadrants))
    low,high=np.quantile(normalized[mask],[.25,.75]);edge=mask&(normalized<=low);core=mask&(normalized>=high)
    for q in (None,.5):
        for k in range(6):
            features.append(float(channels[:,:,k][edge].mean()-channels[:,:,k][core].mean()) if q is None else float(np.quantile(channels[:,:,k][edge],q)-np.quantile(channels[:,:,k][core],q)))
    features.extend([float(mask.mean()),float(np.median(normalized[mask])),float(sat[mask].mean())]);result=np.asarray(features,np.float32)
    assert result.shape==(DIMENSION,) and np.isfinite(result).all()
    return result
