import os
import glob
import subprocess


from tqdm import tqdm
from datetime import datetime
import time
import itertools
import re
import ray

import numpy as np
from math import ceil
import cv2

import ijson
from shapely.geometry import Polygon
from shapely.strtree import STRtree
from shapely.geometry import Point
import openslide

import util.util as util
import util.args as argparser


def get_measurements(json_file, coords, measure_rads_px):
    
    with open(json_file) as f:
        items = ijson.kvitems(f, "nuc")
        tumor_cell_centers = []
        all_cell_centers = []

        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: collecting cells...")
        for k, v in items:
            #if len(tumor_cell_centers) == 100:
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
        if not ray.is_initialized():
            ray_cont = ray.init()
            
        #print(ray.available_resources())
        
        tumor_tree_id = ray.put(tumor_tree)
        cell_tree_id = ray.put(cell_tree)
        coords_id = ray.put(coords)
        
        @ray.remote
        def query_cell_tree(start_id, stop_id, coords, rad, tree):
            values = np.zeros(stop_id-start_id)
            #for i in range(len(coords)):
            for i in range(stop_id-start_id):
                x, y = coords[start_id + i]
                
                cells_in_range = tree.query(Point(x, y).buffer(rad))
                values[i] = len(cells_in_range)
            return values
        
        chunk_size = 10000
        
        
        results = np.zeros((len(coords), len(measure_rads_px), 2))
        
        for j in range(len(measure_rads_px)):
            print(f"computing for rad {measure_rads_px[j]}, tumor cell")
            
            futures = [query_cell_tree.remote(x, np.min((x + chunk_size, len(coords)-1)), coords_id, measure_rads_px[j], tumor_tree_id)
                   for x in range(0, len(coords), chunk_size)]
        
            for i in range(len(futures)):
                results[(i * chunk_size):np.min(((i+1)*chunk_size, len(results)-1)), j, 0] = ray.get(futures[i])
            
            print(f"computing for rad {measure_rads_px[j]}, all cell")

            futures = [query_cell_tree.remote(x, np.min((x + chunk_size, len(coords)-1)), coords_id, measure_rads_px[j], cell_tree_id)
                       for x in range(0, len(coords), chunk_size)]


            for i in range(len(futures)):
                results[(i * chunk_size):np.min(((i+1)*chunk_size, len(results)-1)), j, 1] = ray.get(futures[i])
        #results = ray.get(futures)
        ray.shutdown()
        #subprocess.check_output(["ray", "stop", "--force"])
    return results



def export_measurments_from_jsons(wsi_dir, hover_dir, out_dir, grid_size, grid_in_mu,
                     measure_rads, rad_in_mu):
    
    wsi_path_list = util.get_file_list(wsi_dir)
    center_offset = ceil(grid_size/2)

    for path in wsi_path_list:
        print(path)

        wsi_name = os.path.splitext(os.path.basename(path))[0]

        
        osh = openslide.OpenSlide(path)
        
        grid_size_x, grid_size_y = util.get_grid_size_px_from_openslide(osh, grid_size, grid_in_mu)
        

        if rad_in_mu:
            #ToDo make oval
            measure_rads_px = [ceil(rad/float(osh.properties["openslide.mpp-x"]))
                               for rad in measure_rads]
        else:
            measure_rads_px = measure_rads
            
        


        json_file = hover_dir + "/" + wsi_name + ".json"
        
        coords = list(itertools.product(range(0, osh.dimensions[0], grid_size_x),
                                            range(0, osh.dimensions[1], grid_size_y)))

        results = get_measurements(json_file, coords, measure_rads_px)
        #print(results.shape)
        #print(results)


        dict_out = {}
        for i in range(len(measure_rads_px)):
            dict_out[util.get_npy_coord_name(grid_size, grid_in_mu)] = coords
            dict_out[util.get_npy_measure_name(grid_size, grid_in_mu, measure_rads[i], 
                                               rad_in_mu, "tumor_cells")] = results[:,i,0]
            dict_out[util.get_npy_measure_name(grid_size, grid_in_mu, measure_rads[i], 
                                               rad_in_mu, "all_cells")] = results[:,i,1]
            

        np.savez(f"{out_dir}/{wsi_name}.npz", **dict_out)


        


if __name__ == '__main__':
    #wsi_dir = "/home/vita/Documents/Digital_Pathology/Project/out/ServerRuns/pipe_test/slides_x"
    #hover_dir ="/home/vita/Documents/Digital_Pathology/Project/out/ServerRuns/pipe_test/out_wsi/hover/"
    #out_dir = "../out/qupath_test"
    #measure_dir = "../out/measure"
    #grid_size = 100
    #grid_in_mu = False
    #min_density = 100
    #min_density_in_mu2 = False
    #min_ratio = 0.3

    args = argparser.parse_args()
    
    run_tiles = args.use_tiles
    wsi_dir = args.in_dir
    hover_dir = util.get_hover_dir(args.out_dir)
    measure_dir = util.get_measure_dir(args.out_dir, True)
    
    
    grid_in_mu = False
    grid_size = args.measurement_grid_size_px
    if args.measurement_grid_size_mu:
        grid_size = args.measurement_grid_size_mu
        grid_in_mu = True
        
    
        
    
    rad_in_mu = False
    measure_rads = args.count_rad_px
    if args.count_rad_mu:
        measure_rads = args.count_rad_mu
        rad_in_mu = True
    
    export_measurments_from_jsons(wsi_dir, hover_dir, measure_dir, grid_size,
                                  grid_in_mu, measure_rads, rad_in_mu)
    
    
    
    
        
    
    
    
        
    