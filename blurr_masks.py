# +
import os
import glob

from tqdm import tqdm
from datetime import datetime

from scipy.io import loadmat
import cv2 
import imageio
import numpy as np

import util.util as util
import util.args as argparser


# -
def blurr_masks_flat(in_dir, out_dir, radius, force_rewrite = False):
    mask_files = glob.glob(in_dir+"/*.png")
    
    d = np.array(range(0 - radius, radius + 1))**2
    k = np.tile(d, (len(d), 1))
    k2 = k + k.transpose()
    kernel = (k2 <= radius**2).astype(int)
    kernel = kernel/np.sum(kernel)
    
    for mask_file in tqdm(mask_files):
        name = os.path.basename(mask_file)
        
        blurr_name = out_dir + "/" + name
        if not os.path.exists(blurr_name) or force_rewrite:
            mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        
            mask_blurr = util.convolve(mask, kernel)
            cv2.imwrite(blurr_name, mask_blurr)


def blurr_masks_gauss(in_dir, out_dir, sigma):
    mask_files = glob.glob(in_dir+"/*.png")
    
    for mask_file in tqdm(mask_files):
        name = os.path.basename(mask_file)
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)/255
        
        mask_blurr = util.convolve_gaussian(mask, gauss_sigma)
        cv2.imwrite(out_dir + "/" + name, mask_blurr)


if __name__ == '__main__':
    #setup arguments
    
    args = argparser.parse_args()
    
    wsi_in_dir = args.in_dir
    out_dir = args.out_dir
    
    kernel_radius = args.blurr_flat_rad
    
    #kernel_radius = 200
    #wsi_in_dir = "../data_in"
    #out_dir = "../out"

    #gather files
    path_list = glob.glob(wsi_in_dir + "/*")
    wsi_path_list = [p for p in path_list if os.path.isfile(p)]
    wsi_path_list.sort()

    if len(wsi_path_list) == 0:
        print(f"no files found in {in_dir}.")
    

    #setup out dirs

    
    
    for path in wsi_path_list:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: blurring masks for {os.path.basename(path)}", flush=True)
        wsi_name = os.path.splitext(os.path.basename(path))[0]
        mask_dir_path = out_dir + "/mask/" + wsi_name
        blurr_dir_path = out_dir + f"/blurr_{kernel_radius}/" + wsi_name
        
        if not os.path.exists(blurr_dir_path):
            os.makedirs(blurr_dir_path)
        
        blurr_masks_flat(mask_dir_path, blurr_dir_path, kernel_radius)


