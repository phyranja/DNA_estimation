# +
import os
import glob

from tqdm import tqdm
from datetime import datetime

from paquo.images import QuPathImageType
from paquo.projects import QuPathProject

import cv2
import itertools
import re
import numpy as np
from shapely.geometry import Polygon


# -

def qupath_from_tile_masks(wsi_dir, blurr_tile_dir, out_dir, tile_size, padding, grid_size):

    crop = int(padding/2)
    
    with QuPathProject(out_dir, mode='x') as qp:
    
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Creating QuPath project", qp.name, flush=True)
    
        path_list = glob.glob(wsi_dir + "/*")
        wsi_path_list = [p for p in path_list if os.path.isfile(p)]
        wsi_path_list.sort()
        
        for path in wsi_path_list:
        
            wsi_name = os.path.splitext(os.path.basename(path))[0]
        
            entry = qp.add_image(path, image_type=QuPathImageType.BRIGHTFIELD_H_E)
        
            blurr_files = glob.glob(blurr_tile_dir + wsi_name + "/*.png")
    
    
            for file in tqdm(blurr_files):
                im = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
                hight, width = im.shape
        
            
                mx = re.search('x=(\d+)_', os.path.basename(file))
                offset_x=int(mx.group(1))
                my = re.search('_y=(\d+)_', os.path.basename(file))
                offset_y=int(my.group(1))
                mt = re.search('_ts=(\d+).', os.path.basename(file))
                file_tile_size=int(mt.group(1))
                if file_tile_size != tile_size:
                    continue
        

                for x, y in tqdm(list(itertools.product(range(0, width-padding, grid_size),
                                                                    range(0, hight-padding, grid_size)))):
        
                    if np.mean(im[y+crop:y+crop+grid_size, x+crop:x+crop + grid_size]) > 0:
            
                        tile = Polygon.from_bounds(x + crop + offset_x, y + crop + offset_y,
                                x + crop + offset_x + grid_size, y + crop + offset_y + grid_size)

                        # add tiles (tiles are specialized detection objects drawn without border)

                        detection = entry.hierarchy.add_tile(roi=tile, measurements=
                                                     {'tumor cell density': np.mean(im[y+crop:y+crop+tile_size, x+crop:x+crop + tile_size])})
        
            print("added", len(entry.hierarchy.detections), "tiles for ", os.path.basename(path))
    


if __name__ == '__main__':
    #setup arguments

    run_tiles = True
    in_dir = "../data_in"
    out_dir = "../out"
    
    

    tile_size = 2000
    padding = 500


    pred_gridsize = 200

    kernel_rad = 200
    
    blurr_dir = out_dir + f"/blurr_{kernel_rad}/"
    qupath_out_dir = out_dir + "/qupath"
    if not os.path.exists(qupath_out_dir):
        os.makedirs(qupath_out_dir)
    
    qupath_from_tile_masks(in_dir, blurr_dir, qupath_out_dir, tile_size, padding, pred_gridsize)


