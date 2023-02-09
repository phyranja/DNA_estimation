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
from math import ceil
import ijson
from shapely.geometry import Polygon
from shapely.strtree import STRtree
from shapely.geometry import Point


import util.args as argparser


# +
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
        

                for y, x in itertools.product(range(0, width-padding, grid_size),
                                                range(0, hight-padding, grid_size)):
        
                    if np.mean(im[y+crop:y+crop+grid_size, x+crop:x+crop + grid_size]) > 0:
            
                        tile = Polygon.from_bounds(x + crop + offset_x, y + crop + offset_y,
                                x + crop + offset_x + grid_size, y + crop + offset_y + grid_size)

                        # add tiles (tiles are specialized detection objects drawn without border)

                        detection = entry.hierarchy.add_tile(roi=tile, measurements=
                                                     {'tumor cell density': np.mean(im[y+crop:y+crop+grid_size, x+crop:x+crop + grid_size]/255)})
        
            print("added", len(entry.hierarchy.detections), "tiles for ", os.path.basename(path))
            
            

# +
def qupath_from_json(wsi_dir, hover_dir, out_dir, grid_size, measure_rad):
    
    with QuPathProject(out_dir, mode='w') as qp:
    
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Creating QuPath project", qp.name, flush=True)
    
        path_list = glob.glob(wsi_dir + "/*")
        wsi_path_list = [p for p in path_list if os.path.isfile(p)]
        wsi_path_list.sort()
        center_offset = ceil(grid_size/2)
        
        for path in wsi_path_list:
            print(path)
        
            wsi_name = os.path.splitext(os.path.basename(path))[0]
        
            entry = qp.add_image(path, image_type=QuPathImageType.BRIGHTFIELD_H_E)
        
            json_file = hover_dir + "/json/" + wsi_name + ".json"
            
            
            with open(json_file) as f:
                items = ijson.kvitems(f, "nuc")
                cancer_centers = []
                
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: collecting cells...")
                for k, v in items:
                    if v["type"] == 1:
                        #print(k, v["type"],v["centroid"])
                        cancer_centers.append(Point(v["centroid"]))
                print(f"found {len(cancer_centers)} tumor cells")
                
                
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: building searchtree...")
                tree = STRtree(cancer_centers)
                
                
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: generating measurement")
                
                for x, y in tqdm(list(itertools.product(range(0, entry.width, grid_size),
                                                range(0, entry.height, grid_size)))):
                    
                    cells_in_range = tree.query(Point(x + center_offset, y + center_offset).buffer(measure_rad))
                    num_cells = len(cells_in_range)
        
                    if num_cells > 0:
            
                        tile = Polygon.from_bounds(x, y,
                                x + grid_size, y + grid_size)

                        # add tiles (tiles are specialized detection objects drawn without border)

                        detection = entry.hierarchy.add_tile(roi=tile, measurements=
                                                     {f'tumor cell count within {measure_rad} pixels': num_cells})
        
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: added {len(entry.hierarchy.detections)} tiles for {os.path.basename(path)}")

            

# +
if __name__ == '__main__':
    #setup arguments
    
    args = argparser.parse_args()
    
    run_tiles = args.use_tiles
    in_dir = args.in_dir
    out_dir = args.out_dir
    
    
    
    pred_gridsize = args.measurement_grid_size

    
    
    #in_dir = "/home/vita/Documents/Digital_Pathology/Project/out/ServerRuns/estim_run_1/estim/slides_in"
    #out_dir = "/home/vita/Documents/Digital_Pathology/Project/out/ServerRuns/estim_run_1/estim/out"
    
    qupath_out_dir = out_dir + f"/qupath_{pred_gridsize}"
    if not os.path.exists(qupath_out_dir):
        os.makedirs(qupath_out_dir)
        
    if run_tiles:
        tile_size = args.tile_size
        padding = args.padding_size
        kernel_rad = args.blurr_flat_rad
        
        blurr_dir = out_dir + f"/blurr_{kernel_rad}/"
        qupath_from_tile_masks(in_dir, blurr_dir, qupath_out_dir, tile_size, padding, pred_gridsize)
        
    else:
        measure_rad = args.count_rad
        hover_dir = out_dir + "/hover/"
        qupath_from_json(in_dir, hover_dir, qupath_out_dir, pred_gridsize, measure_rad)   
    
    
    
# -


