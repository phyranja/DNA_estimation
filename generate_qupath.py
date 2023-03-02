# +
import os
import glob


from tqdm import tqdm
from datetime import datetime
import itertools
import re

import numpy as np
from math import ceil
import cv2

from paquo.images import QuPathImageType
from paquo.projects import QuPathProject
import openslide

import ijson
from shapely.geometry import Polygon
from shapely.strtree import STRtree
from shapely.geometry import Point


import util.args as argparser


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
        

                for y, x in itertools.product(range(0, width-padding, grid_size),
                                                range(0, hight-padding, grid_size)):
        
                    if np.mean(im[y+crop:y+crop+grid_size, x+crop:x+crop + grid_size]) > 0:
            
                        tile = Polygon.from_bounds(x + crop + offset_x, y + crop + offset_y,
                                x + crop + offset_x + grid_size, y + crop + offset_y + grid_size)

                        # add tiles (tiles are specialized detection objects drawn without border)

                        detection = entry.hierarchy.add_tile(roi=tile, measurements=
                                                     {'tumor cell density': np.mean(im[y+crop:y+crop+grid_size, x+crop:x+crop + grid_size]/255)})
        
            print("added", len(entry.hierarchy.detections), "tiles for ", os.path.basename(path))



def qupath_from_json(wsi_dir, hover_dir, out_dir, grid_size, grid_in_mu, measure_rads, rad_in_mu):
    
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
            
            if grid_in_mu:
                osh = openslide.OpenSlide(path)
                grid_size_x = ceil(grid_size / float(osh.properties["openslide.mpp-x"]))
                grid_size_y = ceil(grid_size / float(osh.properties["openslide.mpp-y"]))
            else:
                grid_size_x = grid_size
                grid_size_y = grid_size
                
            if rad_in_mu:
                osh = openslide.OpenSlide(path)
                #ToDo make oval
                measure_rads_px = [ceil(rad / float(osh.properties["openslide.mpp-x"]))
                                   for rad in measure_rads]
                detection_text_count = [f'tumor cell count within {rad} microns'
                                        for rad in measure_rads]
                detection_text_ratio = [f'tumor cell ratio within {rad} microns'
                                        for rad in measure_rads]
                max_ratio_text = [f'max tumor cell ratio within {rad} microns'
                                  for rad in measure_rads]
                max_count_text = [f'max tumor cell count within {rad} microns'
                                  for rad in measure_rads]
            else:
                measure_rads_px = measure_rads
                detection_text_count = [f'tumor cell count within {rad} pixels' 
                                        for rad in measure_rads]
                detection_text_ratio = [f'tumor cell ratio within {rad} pixels' 
                                        for rad in measure_rads]
                max_ratio_text = [f'max tumor cell ratio within {rad} pixels'
                                  for rad in measure_rads]
                max_count_text = [f'max tumor cell count within {rad} pixels'
                                  for rad in measure_rads]
                
                
        
            json_file = hover_dir + wsi_name + ".json"
            
            
            with open(json_file) as f:
                items = ijson.kvitems(f, "nuc")
                tumor_cell_centers = []
                all_cell_centers = []
                
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: collecting cells...")
                for k, v in items:
                    #if len(tumor_cell_centers) == 1000:
                    #    break
                    all_cell_centers.append(Point(v["centroid"]))
                    if v["type"] == 1:
                        #print(k, v["type"],v["centroid"])
                        tumor_cell_centers.append(Point(v["centroid"]))
                print(f"found {len(tumor_cell_centers)} tumor cells")
                
                
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: building searchtree...")
                tumor_tree = STRtree(tumor_cell_centers)
                cell_tree = STRtree(all_cell_centers)
                
                
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: generating measurement")
                
                max_tumor_cell_count_coords = np.zeros((len(measure_rads_px), 2))
                max_tumor_cell_count = np.zeros(len(measure_rads_px))
                max_ratio_coords = np.zeros((len(measure_rads_px),2))
                max_ratio = np.zeros(len(measure_rads_px))
                
                for x, y in tqdm(list(itertools.product(range(0, entry.width, grid_size_x),
                                                range(0, entry.height, grid_size_y)))):
                    
                    num_cells_tot = np.zeros(len(measure_rads_px))
                    num_cells_tumor = np.zeros(len(measure_rads_px))
                    tumor_cell_ratio = np.zeros(len(measure_rads_px))
                    for i in range(len(measure_rads_px)):
                        cells_in_range = cell_tree.query(Point(x + center_offset, y + center_offset).buffer(measure_rads_px[i]))
                        num_cells_tot[i] = len(cells_in_range)
                        tumor_cells_in_range = tumor_tree.query(Point(x + center_offset, y + center_offset).buffer(measure_rads_px[i]))
                        num_cells_tumor[i] = len(tumor_cells_in_range)
                        tumor_cell_ratio[i] = num_cells_tumor[i]/num_cells_tot[i] if num_cells_tot[i] else 0
                        
                        
                    if any(num_cells_tumor):
                            
                        
                        tile = Polygon.from_bounds(x, y, x + grid_size_x, y + grid_size_y)
                        
                        tile_measurements = {detection_text_count[i]: num_cells_tumor[i] 
                                       for i in range(len(num_cells_tumor))}
                        
                        tile_measurements.update({detection_text_ratio[i]: tumor_cell_ratio[i]
                                             for i in range(len(num_cells_tumor))})
                        
                        for i in range(len(tumor_cell_ratio)):
                            if tumor_cell_ratio[i] > 0.3 and num_cells_tumor[i] >= 200:
                                tile_measurements[detection_text_ratio[i] + ">0.3, min 200 cells"] = tumor_cell_ratio[i]
                                
                                if max_tumor_cell_count[i] < num_cells_tumor[i]:
                                    max_tumor_cell_count[i] = num_cells_tumor[i]
                                    max_tumor_cell_count_coords[i] = [x + center_offset, y + center_offset]
                                if max_ratio[i] < tumor_cell_ratio[i]:
                                    max_ratio[i] = tumor_cell_ratio[i]
                                    max_ratio_coords[i] = [x + center_offset, y + center_offset]
                        
                        tile_measurements.update({"total cell count" + str(i): num_cells_tot[i]
                                             for i in range(len(num_cells_tumor))})
                    

                        # add tiles (tiles are specialized detection objects drawn without border)
                        detection = entry.hierarchy.add_tile(roi=tile, measurements=
                                                     tile_measurements)
                        #detection = entry.hierarchy.add_tile(roi=Point(x + center_offset, y + center_offset).buffer(measure_rads_px), measurements=
                        #                             {"yy" : num_cells_tot[0]})
                for i in range(len(measure_rads_px)):
                    entry.hierarchy.add_tile(roi=Point(max_ratio_coords[i]).buffer(measure_rads_px[i]), measurements=
                                            {max_ratio_text[i] : max_ratio[i]})
                    entry.hierarchy.add_tile(roi=Point(max_tumor_cell_count_coords[i]).buffer(measure_rads_px[i]), measurements=
                                            {max_count_text[i] : max_tumor_cell_count[i]})
                        
            
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: added {len(entry.hierarchy.detections)} tiles for {os.path.basename(path)}")





# +
if __name__ == '__main__':
    #setup arguments
    
    args = argparser.parse_args()
    
    run_tiles = args.use_tiles
    in_dir = args.in_dir
    out_dir = args.out_dir
    
    gridsize_in_mu = False
    pred_gridsize = args.measurement_grid_size_px
    if args.measurement_grid_size_mu:
        pred_gridsize = args.measurement_grid_size_mu
        gridsize_in_mu = True
        
        
    print(pred_gridsize, gridsize_in_mu)

    
    
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
        rad_in_mu = False
        measure_rad = args.count_rad_px
        if args.count_rad_mu:
            measure_rad = args.count_rad_mu
            rad_in_mu = True
        
        hover_dir = out_dir + "/hover/"
        qupath_from_json(in_dir, hover_dir, qupath_out_dir, pred_gridsize, gridsize_in_mu,
                         measure_rad, rad_in_mu)   
    
    
    
# -


