"""Label-symmetric RGB lighting variation; clinical labels are never changed."""
import torch


def lighting_variation(rgb,brightness,contrast,saturation,white_balance):
    # NCHW RGB01; all factors are N-by-1-by-1-by-1 except N-by-3 white balance.
    gray=(rgb*rgb.new_tensor([.299,.587,.114])[None,:,None,None]).sum(1,keepdim=True)
    result=gray+saturation*(rgb-gray)
    mean=result.mean((1,2,3),keepdim=True);result=mean+contrast*(result-mean)
    return (result*brightness*white_balance[:,:,None,None]).clamp(0,1)
