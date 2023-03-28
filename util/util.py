import torch
from tqdm import tqdm
import openslide
import cv2
import os
import glob
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
                cv2.imwrite(tile_name, cv2.cvtColor(asnumpy(tile), cv2.COLOR_RGB2BGR))
                del tile

                
def get_file_list(file_dir):
    path_list = glob.glob(file_dir + "/*")
    file_path_list = [p for p in path_list if os.path.isfile(p)]
    file_path_list.sort()
    return file_path_list



### convert microns to pixels if needed:

#gridsize from microns per pixel
def get_grid_size_px(grid_size, grid_in_mu, mpp_x, mpp_y):
    
    if grid_in_mu:
        grid_size_x = math.ceil(grid_size /float(mpp_x))
        grid_size_y = math.ceil(grid_size / float(mpp_y))
    else:
        grid_size_x = grid_size
        grid_size_y = grid_size
        
    return (grid_size_x, grid_size_y)

#gridsize from a QuPath image entry
def get_grid_size_px_from_qupath(qupath_entry, grid_size, grid_in_mu):
    
    return get_grid_size_px(grid_size, grid_in_mu,
                            float(qupath_entry._image_server.getMetadata().getPixelWidthMicrons()),
                            float(qupath_entry._image_server.getMetadata().getPixelHeightMicrons()))

#gridsize from an Openslide handel
def get_grid_size_px_from_openslide(osh, grid_size, grid_in_mu):
    return get_grid_size_px(grid_size, grid_in_mu,
                            float(osh.properties["openslide.mpp-x"]),
                            float(osh.properties["openslide.mpp-y"]))



### get output directories

def get_hover_dir(base_dir, make_dir = False):
    hover_dir = base_dir.rstrip("/") + "/hover"
    if make_dir and not os.path.exists(hover_dir):
        os.makedirs(hover_dir)
    return hover_dir
        
def get_measure_dir(base_dir, make_dir = False):
    measure_dir = base_dir.rstrip("/") + "/measure"
    if make_dir and not os.path.exists(measure_dir):
        os.makedirs(measure_dir)
    return measure_dir

def get_qupath_dir(base_dir, grid_size, grid_size_in_mu, make_dir = False):
    qupath_dir = base_dir.rstrip("/") + "/qupath_" + str(grid_size)
    if make_dir and not os.path.exists(qupath_dir):
        os.makedirs(qupath_dir)  
    return qupath_dir

### get file names and other strings

def get_npy_measure_name(grid_size, grid_size_in_mu, radius, rad_in_mu, prefix):
    return f"{prefix}_grid={grid_size}{'mu' if grid_size_in_mu else 'px'}_rad={radius}{'mu' if rad_in_mu else 'px'}"

def get_npy_coord_name(grid_size, grid_size_in_mu):
    return f"coords_grid={grid_size}{'mu' if grid_size_in_mu else 'px'}"