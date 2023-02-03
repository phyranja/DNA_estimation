import torch
from tqdm import tqdm
import openslide
from cv2 import imwrite
import os
import itertools
import math

if(torch.cuda.is_available()):
    import cupy as xp
    import cupyx.scipy as sp
    from cupyx.scipy.signal import convolve2d
    from cupyx.scipy.ndimage import gaussian_filter
    from cupy import asnumpy
    
else:
    import numpy as xp
    import scipy as sp
    from scipy.signal import convolve2d
    from scipy.ndimage import gaussian_filter
    from numpy import asarray as asnumpy


def hover_accumulate_instance_masks_area(inst_map, id_list):
    xp_map = xp.asarray(inst_map) #convert to cupy array if used
    
    mask = xp.zeros(inst_map.shape)

    for idx in id_list:
        mask = xp.logical_or(mask, xp_map == idx)
        
    return asnumpy(mask)

def hover_accumulate_instance_masks_center(center_list, out_shape, id_list):
    #convert to cupy array if used
    mask = xp.zeros(out_shape)

    for idx in id_list:
        center = center_list[idx-1]
        mask[math.floor(center[1]), math.floor(center[0])] = 1
        
    return asnumpy(mask)

def convolve(img_in, kernel):
    kernel_xp = xp.asarray(kernel)
    out = xp.asarray(img_in).astype(xp.float64)
    out = convolve2d(out, kernel_xp, mode = 'same')
    return asnumpy(out)
        
def convolve_gaussian(img_in, sigma):
    out = xp.asarray(img_in).astype(xp.float64)
    out = gaussian_filter(out, sigma)
    return asnumpy(out)



def get_tile(osh, openslide_level, tile_size,  padding, y, x):
    tile = osh.read_region((x - paddingsize, y - paddingsize), openslidelevel,
                           (tilesize + 2 * paddingsize, tilesize + 2 * paddingsize))
    return tile

    
    
    
def save_wsi_tiles(osh, tile_size, padding, save_folder, force_rewrite = False):
    #save folder needs to identify wsi, only coordinates are saved in filename
    nrow,ncol = osh.level_dimensions[0]

    for i, j in tqdm(list(itertools.product(range(0, ncol, tile_size),
                                  range(0, nrow, tile_size)))):
#    for i in range(0, ncol, tile_size):
#        for j in range(0, nrow, tile_size):
            tile_name = f"{save_folder}/x={j}_y={i}_ts={tile_size}.png"
            if not os.path.exists(tile_name) or force_rewrite:
                tile = osh.read_region((j, i), 0, (tile_size + padding,
                                                   tile_size + padding))
                imwrite(tile_name, asnumpy(tile))
                del tile
