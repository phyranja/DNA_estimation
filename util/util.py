import torch
from tqdm.autonotebook import tqdm

if(torch.cuda.is_available()):
    import cupy as xp
    import cupyx.scipy as sp
    from cupyx.scipy.signal import convolve2d
    from cupyx.scipy.ndimage import gaussian_filter
    
else:
    import numpy as xp
    import scipy as sp
    from scipy.signal import convolve2d
    from scipy.ndimage import gaussian_filter


def hover_accumulate_instance_masks(inst_map, id_list):
    mask = xp.zeros(inst_map.shape)

    for idx in tqdm(id_list):
        mask = xp.logical_or(mask, inst_map == idx)
        
    return mask

def convolve_iter(img_in, kernel, iterations):
    out = img_in.astype(xp.float64)
    for i in tqdm(range(iterations)):
        out = convolve2d(out, kernel, mode = 'same')
    
    return out
        
def convolve_gaussian_iter(img_in, sigma, iterations):
    out = img_in.astype(xp.float64)
    for i in tqdm(range(iterations)):
        out = gaussian_filter(out, sigma)
    
    return out