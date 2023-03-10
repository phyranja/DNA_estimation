# +
import os
import glob

from tqdm import tqdm
from datetime import datetime

from scipy.io import loadmat
from cv2 import imread
from cv2 import imwrite
import numpy as np

import util.util as util
import util.args as argparser
# -





def generate_masks(in_dir, out_dir, type_id, force_rewrite = False):
    mat_files = glob.glob(in_dir+"/*.mat")
    
    for mat_file in tqdm(mat_files):
        
        name = os.path.basename(mat_file)
        mask_name = out_dir + "/" + name +".png"
        if not os.path.exists(mask_name) or force_rewrite:
            mat = loadmat(mat_file)
        
            #create mask of all cancer cells
            cancer_ids = [ mat["inst_uid"][i][0] for i in range(len(mat["inst_type"])) if mat["inst_type"][i] == type_id]
            mask = util.hover_accumulate_instance_masks_area(mat["inst_map"], cancer_ids)
            imwrite(mask_name, mask.astype(np.uint8)*255)



if __name__ == '__main__':
    #setup arguments
    
    args = argparser.parse_args()
    
    wsi_in_dir = args.in_dir
    out_dir = args.out_dir

    hover_class_id = args.hover_class

    #gather files
    path_list = glob.glob(wsi_in_dir + "/*")
    wsi_path_list = [p for p in path_list if os.path.isfile(p)]
    wsi_path_list.sort()

    if len(wsi_path_list) == 0:
        print(f"no files found in {in_dir}.")
    

    #setup out dirs

    
    
    for path in wsi_path_list:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: generating masks for {os.path.basename(path)}", flush=True)
        wsi_name = os.path.splitext(os.path.basename(path))[0]
        hover_out_path = out_dir + "/hover/" + wsi_name +"/mat"
        mask_dir_path = out_dir + "/mask/" + wsi_name
        
        if not os.path.exists(mask_dir_path):
            os.makedirs(mask_dir_path)
        
        generate_masks(hover_out_path, mask_dir_path, hover_class_id)


