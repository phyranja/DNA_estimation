import torch
from tqdm.autonotebook import tqdm

if(torch.cuda.is_available()):
    import cupy as np
    import cupyx.scipy as sp
    from cupyx.scipy.signal import convolve2d
    
else:
    import numpy as np
    import scipy as sp
    from scipy.signal import convolve2d


def hover_accumulate_instance_masks(hover_mat, id_list):
    mask = np.zeros(hover_mat["inst_map"].shape)
    inst_map = np.asarray(hover_mat["inst_map"])

    for idx in tqdm(id_list):
        mask = np.logical_or(mask, inst_map == idx)
        
    return mask

def convolve_iter(img_in, kernel, iterations):
    out = img_in
    for i in tqdm(range(iterations)):
        out = convolve2d(out, kernel, mode = 'same')
    
    return out
        
    