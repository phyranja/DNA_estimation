# +
import os
import glob
import subprocess

from tqdm import tqdm
import re
from datetime import datetime

import openslide
from paquo.images import QuPathImageType
from paquo.projects import QuPathProject

from scipy.io import loadmat
from imageio import imread
from imageio import imwrite
import numpy as np
from scipy.ndimage import generate_binary_structure

import util.util as util
from extract_tiles import extract_tiles_wsi_dir
from generate_masks import generate_masks
from blurr_masks import blurr_masks_flat
from generate_qupath import qupath_from_tile_masks

# +
#setup arguments

run_tiles = True
#in_dir = "/home/vita/Documents/Digital_Pathology/Project/data/slide_in_test"
#out_dir = "/home/vita/Documents/Digital_Pathology/Project/out/estim_test_out"
in_dir = "../data_in"
out_dir = "../out"


save_masks = True
save_blurred_masks = True

tile_size = 2000
padding = 500


pred_gridsize = 200


gpu_id = '0,1'
hover_batch_size = 64
hover_model_path = "../checkpoints/hovernet_fast_pannuke_type_tf2pytorch.tar"
hover_infer_workers = 4
hover_post_worers = 4

hover_num_classes = 6
hover_class = 1 # cell type of interest

kernel_rad = 100
d = np.array(range(0 - kernel_rad, kernel_rad + 1))**2
k = np.tile(d, (len(d), 1))
k2 = k + k.transpose()
kernel = (k2 <= kernel_rad**2).astype(int)


# +
#gather files
path_list = glob.glob(in_dir + "/*")
wsi_path_list = [p for p in path_list if os.path.isfile(p)]
wsi_path_list.sort()

if len(wsi_path_list) == 0:
    print(f"no files found in {in_dir}.")
    

#setup out dirs

if save_masks and not os.path.exists(out_dir+"/mask"):
    os.makedirs(out_dir+"/mask")
if save_blurred_masks and not os.path.exists(out_dir+"/blurr"):
    os.makedirs(out_dir+"/blurr")
if not os.path.exists(out_dir+"/qupath"):
    os.makedirs(out_dir+"/qupath")
    

# -

#save tiles if applicable
if run_tiles:
    extract_tiles_wsi_dir(in_dir, out_dir, tile_size, padding)

#run hovernet inference
#todo, check if already exists
if run_tiles:
    for path in wsi_path_list:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: running inference on tiles for {os.path.basename(path)}", flush=True)
        wsi_name = os.path.splitext(os.path.basename(path))[0]
        tile_dir_path = out_dir + "/tiles/" + wsi_name
        hover_dir_path = out_dir + "/hover/" + wsi_name
        
            
        hover_command = ["python", "hover_net/run_infer.py",
        f"--gpu='{gpu_id}'",
        f"--nr_types={hover_num_classes}",
        f"--type_info_path=hover_net/type_info.json",
        f"--batch_size={hover_batch_size}",
        f"--model_mode=fast",
        f"--model_path={hover_model_path}",
        f"--nr_inference_workers={hover_infer_workers}",
        f"--nr_post_proc_workers={hover_post_worers}",
        f"tile",
        f"--input_dir={tile_dir_path}",
        f"--output_dir={hover_dir_path}",
        f"--mem_usage=0.1 "]
        
        with subprocess.Popen(hover_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as proc:
            for line in proc.stderr:
                print("hover:", line)



#setup instance masks
for path in wsi_path_list:
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: generating masks for {os.path.basename(path)}", flush=True)
    wsi_name = os.path.splitext(os.path.basename(path))[0]
    hover_out_path = out_dir + "/hover/" + wsi_name + "/mat"
    mask_dir_path = out_dir + "/mask/" + wsi_name
    
    generate_masks(hover_out_path, mask_dir_path, hover_class)

# blurr masks
for path in wsi_path_list:
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: blurring masks for {os.path.basename(path)}", flush=True)
    wsi_name = os.path.splitext(os.path.basename(path))[0]
    mask_dir_path = out_dir + "/mask/" + wsi_name
    blurr_dir_path = out_dir + f"/blurr_{kernel_rad}/" + wsi_name
    
    if not os.path.exists(blurr_dir_path):
            os.makedirs(blurr_dir_path)
            
    blurr_masks_flat(mask_dir_path, blurr_dir_path, kernel_rad)
            


blurr_dir = out_dir + f"/blurr_{kernel_rad}/"
qupath_out_dir = out_dir + "/qupath"
if not os.path.exists(qupath_out_dir):
    os.makedirs(qupath_out_dir)
qupath_from_tile_masks(in_dir, blurr_dir, qupath_out_dir, tile_size, padding, pred_gridsize)


