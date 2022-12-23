# +
import matplotlib.pyplot as plt
from imageio import imread
from shapely.geometry import Polygon
from paquo.images import QuPathImageType
from paquo.projects import QuPathProject
import numpy as np

import os
import glob

import itertools
from typing import Tuple, Iterator
from tqdm.autonotebook import tqdm


# -

def iterate_grid(width, height, step) -> Iterator[Tuple[int, int]]:

    yield from itertools.product(

        range(0, width, step),

        range(0, height, step)

    )


# +
in_dir = "/home/vita/Documents/Digital_Pathology/Project/out/ServerRuns/conv_out_tiles_42/"
wsi_file = "/home/vita/Documents/Digital_Pathology/Project/data/Slides/TCGA-42-2590-01Z-00-DX1.83ff5df9-0f36-4884-a93e-f2928ff3c719.svs"
out_dir = "/home/vita/Documents/Digital_Pathology/Project/code/out_dir/QuPath_tile_3"


im_files = glob.glob(in_dir+"blurr/*.mat_gauss_200.png")

tile_size = 500
im_padding = 200
crop = int(im_padding/2)


# +


with QuPathProject(out_dir, mode='x') as qp:
    
    

    print("created project", qp.name)

    entry = qp.add_image(wsi_file, image_type=QuPathImageType.BRIGHTFIELD_H_E)
    
    
    for file in tqdm(im_files):
        im = imread(file)
        hight, width = im.shape
        
        #read offset from file names -> to be generalized forpipeline
        split_dot = os.path.basename(file).split(".")
        split_u = split_dot[len(split_dot)-3].split("_")

        offset_x=int(split_u[-2])
        offset_y=int(split_u[-1])
        

        for x, y in iterate_grid(width-im_padding, hight-im_padding, tile_size):
        
            if np.mean(im[y+crop:y+crop+tile_size, x+crop:x+crop + tile_size]) > 0.5:
            
                tile = Polygon.from_bounds(x + crop + offset_x, y + crop + offset_y,
                                   x + crop + offset_x + tile_size, y + crop + offset_y + tile_size)

                # add tiles (tiles are specialized detection objects drawn without border)

                detection = entry.hierarchy.add_tile(roi=tile, measurements=
                                                     {'tumor cell density': np.mean(im[y+crop:y+crop+tile_size, x+crop:x+crop + tile_size])})
        
    print("added", len(entry.hierarchy.detections), "tiles")


# +
f = "/home/vita/Documents/Digital_Pathology/Project/out/ServerRuns/conv_out_tiles_42/blurr/TCGA-42-2590-01Z-00-DX1.83ff5df9-0f36-4884-a93e-f2928ff3c719_0_15000.mat_gauss_100.png"
fn = os.path.basename(f)
split_dot = fn.split(".")
split_u = split_dot[len(split_dot)-3].split("_")

print(split_dot)
print(split_u)
offset_x=int(split_u[-2])
offset_y=int(split_u[-1])
print(offset_x, offset_y)
# -



