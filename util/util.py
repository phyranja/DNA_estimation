import torch
from tqdm.autonotebook import tqdm

if(torch.cuda.is_available()):
    import cupy as xp
    import cupyx.scipy as sp
    from cupyx.scipy.signal import convolve2d
    
else:
    import numpy as xp
    import scipy as sp
    from scipy.signal import convolve2d


def hover_accumulate_instance_masks(inst_map, id_list):
    mask = xp.zeros(inst_map.shape)

    for idx in tqdm(id_list):
        mask = xp.logical_or(mask, inst_map == idx)
        
    return mask

def convolve_iter(img_in, kernel, iterations):
    out = img_in.astype(xp.float64)
    for i in tqdm(range(iterations)):
        print(type(out), type(kernel))
        print(out.dtype, kernel.dtype)
        out = convolve2d(out, kernel, mode = 'same')
    
    return out
        
    