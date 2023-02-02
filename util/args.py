from configargparse import ArgParser


def parse_args():
    
    parser = ArgParser()
    
    #possibility to enter arguments via config file
    parser.add_argument('-c', '--config', is_config_file=True, help='config file path')
    
    #IO
    parser.add_argument('--in_dir', help="input directory of WSI files",
                        required=True, type=str)
    parser.add_argument('--out_dir', help="output directory",
                        required=True, type=str)
    
    #tiles
    parser.add_argument('--tile_size', help="pixel size of tiles computed on",
                        default=2000, type=int)
    parser.add_argument('--padding_size', help="size of padding used for tiles",
                        default=200, type=int)
    
    #hover
    parser.add_argument('--gpu_id', help="CUDA id of GPUs to use",
                        default=0, type=int)
    parser.add_argument('--batch_size', help="batch size per GPU (to large batch sizes can lead to memory issues)",
                        default=16, type=int)
    parser.add_argument('--model_path', help="path to hovernet model to use",
                        default="../checkpoints/hovernet_fast_pannuke_type_tf2pytorch.tar", type=str)
    parser.add_argument('--nr_inference_workers', help="Number of workers for inference",
                        default=2, type=int)
    parser.add_argument('--nr_post_proc_workers', help="Number of workers for postprocessing",
                        default=2, type=int)
    parser.add_argument('--mem_usage', help="percentage of memory each hovernet worker is allowed to use",
                        default=0.1, type=float)
    
    #mask generation
    
    parser.add_argument('--hover_class', help="Id of hovernet class of interest",
                        default=1, type=int)
    
    #blurring
    parser.add_argument('--blurr_flat_rad', help="pixel radius of flat kernel",
                        default=200, type=int)
    
    
    #QuPath
    parser.add_argument('--measurement_grid_size', help="gidsize for output measurement tiles",
                        default=500, type=int)
    
    args_parsed = parser.parse_args()
    return args_parsed

if __name__ == '__main__':
    #setup arguments

    x = parse_args()
    print(x)
