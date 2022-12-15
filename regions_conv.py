

import matplotlib.pyplot as plt
import openslide
import os
import glob
import torch
from scipy.io import loadmat
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
if(torch.cuda.is_available()):
    import cupy as np
    import cupy.scipy as sp
    from cupy.scipy.signal import convolve2d
    
else:
    import numpy as np
    import scipy as sp
    from scipy.signal import convolve2d

# +
in_dir = "../in_dir/"
out_dir = "../out_dir/"

kernel_size = 100
threshold = 0.1

if not os.path.exists(out_dir+"mask"):
    os.makedirs(out_dir+"mask")
if not os.path.exists(out_dir+"blurr"):
    os.makedirs(out_dir+"blurr")
if not os.path.exists(out_dir+"json"):
    os.makedirs(out_dir+"json")


mat_files = glob.glob(in_dir+"mat/*.mat")


# +
kernel = np.ones((kernel_size,kernel_size))/(kernel_size*kernel_size)

print(len(mat_files), "files to process")
for mat_file in mat_files:
    name = os.path.basename(mat_file)
    print("processing", name)
    mat = loadmat(mat_file)
    
    cancer_ids = [ mat["inst_uid"][i][0] for i in range(len(mat["inst_type"])) if mat["inst_type"][i] == 1]
    mask = accumulate_masks(mat, cancer_ids)
    cv2.imwrite(out_dir + "mask/" + name + ".png", mask.astype(np.uint8)*255)
    
    mask_blurr = convolve_iter(mask, kernel, 2)
    cv2.imwrite(out_dir + "blurr/" + name + ".png", mask_blurr*255)
    
    mask_regions = mask_blurr > threshold
    cv2.imwrite(out_dir + f"blurr/{name}_t={threshold}.png", mask_blurr*255)
    
    all_polygons = []
    plt.imshow(mask_regions)
    for s, value in features.shapes(mask_regions.astype(np.int16), mask_regions):
        poly = shape(s)
        all_polygons.append(poly)
        x,y = poly.exterior.xy
        plt.plot(x,y)

    json_dicts = []

    for poly in all_polygons:
        json_dicts.append({
            "type": "Feature",
            "geometry": mapping(poly)})
    
    with open(out_dir + f"json/{name}_t={threshold}.json", 'w') as outfile:
        geojson.dump(json_dicts,outfile)

# -


