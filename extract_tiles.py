# +
import os
import glob

from tqdm import tqdm
import re
from datetime import datetime

import openslide
import util.util as util
import util.args as argparser


# -

def extract_tiles_wsi(wsi_path, out_dir, tile_size, pad_size):
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: generating tiles for {os.path.basename(wsi_path)}")
    osh = openslide.OpenSlide(wsi_path)
    wsi_name = os.path.splitext(os.path.basename(wsi_path))[0]
    tile_dir_path = out_dir + "/tiles/" + wsi_name
        
    if not os.path.exists(tile_dir_path):
        os.makedirs(tile_dir_path)
            
    #todo, check if already exists, if yes, skip
    util.save_wsi_tiles(osh, tile_size, pad_size, tile_dir_path)


def extract_tiles_wsi_dir(in_dir, out_dir, tile_size, pad_size):
    #gather files
    path_list = glob.glob(in_dir + "/*")
    wsi_path_list = [p for p in path_list if os.path.isfile(p)]
    wsi_path_list.sort()

    if len(wsi_path_list) == 0:
        print(f"no files found in {in_dir}.")
        return
    
    print(f"Found {len(wsi_path_list)} files to process")

    #setup out dir
    for path in wsi_path_list:
        extract_tiles_wsi(path, out_dir, tile_size, pad_size)


if __name__ == '__main__':
    #setup arguments

    #in_dir = "../data_in"
    #out_dir = "../out"
    #tile_size = 2000
    #padding = 500
    
    args = argparser.parse_args()
    
    in_dir = args.in_dir
    out_dir = args.out_dir
    
    tile_size = args.tile_size
    padding = args.padding_size

    
    extract_tiles_wsi_dir(in_dir, out_dir, tile_size, padding)


