

import openslide
import os
import glob
import torch
from scipy.io import loadmat
import numpy as np
import cv2
from tqdm.autonotebook import tqdm
import time
from rasterio import features
from shapely.geometry import shape
from shapely.geometry import mapping
import geojson
from util.util import hover_accumulate_instance_masks as accumulate_masks
from util.util import convolve_iter

# +
#not the most elegant way, should do it without torch
if(torch.cuda.is_available()):
    import cupy as xp
    import cupyx.scipy as sp
    from cupyx.scipy.signal import convolve2d
    from cupy import asnumpy
    
else:
    import numpy as xp
    import scipy as sp
    from scipy.signal import convolve2d
    from numpy import asarray as asnumpy

# +
#TODO: make these arguments
#in_dir contents expected to be HoverNet output
in_dir = "../in_dir/"
out_dir = "../out_dir/"

kernel_size = 100
threshold = 0.1
inst_type = 1 #type of cells of interesst

#preparing output directories
if not os.path.exists(out_dir+"mask"):
    os.makedirs(out_dir+"mask")
if not os.path.exists(out_dir+"blurr"):
    os.makedirs(out_dir+"blurr")
if not os.path.exists(out_dir+"json"):
    os.makedirs(out_dir+"json")


mat_files = glob.glob(in_dir+"mat/*.mat")


# +
#TODO: test different types of kernel, esp. gaussian
kernel = xp.ones((kernel_size,kernel_size))/(kernel_size*kernel_size)

print(len(mat_files), "files to process")
for mat_file in mat_files:
    name = os.path.basename(mat_file)
    print("processing", name)
    mat = loadmat(mat_file)
    
    #create mask of all cancer cells
    cancer_ids = [ mat["inst_uid"][i][0] for i in range(len(mat["inst_type"])) if mat["inst_type"][i] == inst_type]
    inst_map = xp.asarray(mat["inst_map"]) #convert to cupy array if cupy is used
    
    mask = accumulate_masks(inst_map, cancer_ids)
    cv2.imwrite(out_dir + "mask/" + name + ".png", asnumpy(mask).astype(np.uint8)*255)
    
    mask_blurr = convolve_iter(mask, kernel, 2)
    cv2.imwrite(out_dir + "blurr/" + name + ".png", asnumpy(mask_blurr)*255)
    
    mask_regions = mask_blurr > threshold
    cv2.imwrite(out_dir + f"blurr/{name}_t={threshold}.png", asnumpy(mask_regions)*255)
    
    #create polygons from mask and write them to QuPath compatile json
    all_polygons = []
    np_mask_regions = asnumpy(mask_regions)
    for s, value in features.shapes(np_mask_regions.astype(np.uint8), mask_regions):
        poly = shape(s)
        all_polygons.append(poly)

    json_dicts = []

    for poly in all_polygons:
        json_dicts.append({
            "type": "Feature",
            "geometry": mapping(poly)})
    
    with open(out_dir + f"json/{name}_t={threshold}.json", 'w') as outfile:
        geojson.dump(json_dicts,outfile)

# -


