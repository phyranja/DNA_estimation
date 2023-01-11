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

import util.util as util

# +
#setup arguments

run_tiles = True
#in_dir = "/home/vita/Documents/Digital_Pathology/Project/data/slide_in_test"
#out_dir = "/home/vita/Documents/Digital_Pathology/Project/out/estim_test_out"
in_dir = "../data_in"
out_dir = "../out"


save_masks = True
save_blurred_masks = True

tile_size = 5000
padding = 500


pred_gridsize = 200


gpu_id = '0,1'
hover_batch_size = 64
hover_model_path = "../checkpoints/hovernet_fast_pannuke_type_tf2pytorch.tar"
hover_infer_workers = 4
hover_post_worers = 4

hover_num_classes = 6
hover_class = 1 # cell type of interest


# +
#gather files
wsi_path_list = glob.glob(in_dir + "/*")
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
    for path in wsi_path_list:
        print(f"{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: generating tiles for {os.path.basename(path)}")
        osh = openslide.OpenSlide(path)
        wsi_name = os.path.splitext(os.path.basename(path))[0]
        tile_dir_path = out_dir + "/tiles/" + wsi_name
        
        if not os.path.exists(tile_dir_path):
            os.makedirs(tile_dir_path)
            
        #todo, check if already exists, if yes, skip
        util.save_wsi_tiles(osh, tile_size, padding, tile_dir_path)

#run hovernet inference
#todo, check if already exists
if run_tiles:
    for path in wsi_path_list:
        print(f"{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: running inference on tiles for {os.path.basename(path)}")
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



#setup instance masks and blurred masks
for path in wsi_path_list:
    print(f"{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: generating and blurring masks for {os.path.basename(path)}")
    wsi_name = os.path.splitext(os.path.basename(path))[0]
    hover_out_path = out_dir + "/hover/" + wsi_name
    mat_files = glob.glob(hover_out_path+"/mat/*.mat")
    

    print("Accumulating masks for "+ wsi_name)
    mask_dir_path = out_dir + "/mask/" + wsi_name
    blurr_dir_path = out_dir + "/blurr/" + wsi_name
    
    for mat_file in tqdm(mat_files):
        name = os.path.basename(mat_file)
        mat = loadmat(mat_file)
        
        
        #create mask of all cancer cells
        cancer_ids = [ mat["inst_uid"][i][0] for i in range(len(mat["inst_type"])) if mat["inst_type"][i] == hover_class]
        mask = accumulate_masks(inst_map, cancer_ids)
        cv2.imwrite(mask_dir_path + "/" + name, mask.astype(np.uint8)*255)
    
        if kernel_type == "flat":
            mask_blurr = convolve_iter(mask, kernel, 2)
            cv2.imwrite(blurr_dir_path + f"/{name}_conv_{kernel_size}.png", mask_blurr*255)

        else:
            mask_blurr = convolve_gaussian_iter(mask, gauss_sigma, 2)
            cv2.imwrite(blurr_dir_path + f"/{name}_gauss_{gauss_sigma}.png", mask_blurr*255)
            


# +
import itertools
from typing import Tuple, Iterator
from tqdm.autonotebook import tqdm

def iterate_grid(width, height, step) -> Iterator[Tuple[int, int]]:

    yield from itertools.product(

        range(0, width, step),

        range(0, height, step)

    )


crop = int(padding/2)



with QuPathProject(out_dir + "/qupath", mode='x') as qp:
    
    

    print(f"{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: Creating QuPath project", qp.name)
    
    for path in wsi_path_list:
        
        wsi_name = os.path.splitext(os.path.basename(path))[0]
        
        entry = qp.add_image(path, image_type=QuPathImageType.BRIGHTFIELD_H_E)
        
        blurr_files = glob.glob(out_dir + "/blurr/" + wsi_name + "/*.png")
    
    
        for file in tqdm(blurr_files):
            im = imread(file)
            hight, width = im.shape
        
            
            mx = re.search('x=(\d+)_', os.path.basename(file))
            offset_x=int(mx.group(1))
            my = re.search('_y=(\d+)_', os.path.basename(file))
            offset_y=int(my.group(1))
            mt = re.search('_ts=(\d+).', os.path.basename(file))
            file_tile_size=int(mt.group(1))
            if file_tile_size != tile_size:
                continue
        

            for x, y in iterate_grid(width-padding, hight-padding, tile_size):
        
                if np.mean(im[y+crop:y+crop+tile_size, x+crop:x+crop + tile_size]) > 0:
            
                    tile = Polygon.from_bounds(x + crop + offset_x, y + crop + offset_y,
                                x + crop + offset_x + tile_size, y + crop + offset_y + tile_size)

                    # add tiles (tiles are specialized detection objects drawn without border)

                    detection = entry.hierarchy.add_tile(roi=tile, measurements=
                                                     {'tumor cell density': np.mean(im[y+crop:y+crop+tile_size, x+crop:x+crop + tile_size])})
        
        print("added", len(entry.hierarchy.detections), "tiles for ", os.path.basename(path))
print(f"{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: done")

