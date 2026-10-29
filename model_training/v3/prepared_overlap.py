"""Small geometry packets equivalent to the existing full-image SIFT audit."""
import cv2,numpy as np

def prepare_gray(encoded,mirror=False):
    image=cv2.imdecode(np.frombuffer(encoded,np.uint8),cv2.IMREAD_GRAYSCALE|cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:raise ValueError('Cannot decode source pixels')
    if mirror:image=np.fliplr(image)
    factor=600/max(image.shape)
    return cv2.resize(image,None,fx=factor,fy=factor,interpolation=cv2.INTER_AREA if factor<1 else cv2.INTER_LINEAR)

def prepared_features(image):
    image=np.asarray(image)
    if image.ndim!=2 or image.dtype!=np.uint8 or max(image.shape)!=600:raise ValueError('Canonical600px uint8 grayscale required')
    points,descriptors=cv2.SIFT_create(nfeatures=512).detectAndCompute(image,None)
    if descriptors is None:return np.empty((0,2),np.float32),np.empty((0,128),np.float32),image.shape
    order=sorted(range(len(points)),key=lambda i:(-points[i].response,points[i].pt[0],points[i].pt[1],points[i].angle))[:512]
    return np.array([points[i].pt for i in order],np.float32),descriptors[order],image.shape
