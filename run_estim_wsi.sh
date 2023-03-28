#!/bin/bash

#extract in_dir and out_dir from args for hovernet inference
i=1
for arg in "$@" 
do
    i=$((i + 1));
    case "$arg" in
    --in_dir)
      in_dir="${!i}"
      ;;
    --out_dir)
      out_dir=${!i}
      ;;
    esac 
done


#this should result in the same directory as util.util.get_hover_dir
hover_out_dir="$out_dir"/hover/

#: <<'END'
python hover_net/run_infer.py \
        --gpu=0 \
        --nr_types=6 \
        --type_info_path=hover_net/type_info.json \
        --batch_size=16 \
        --model_mode=fast \
        --model_path=../checkpoints/hovernet_fast_pannuke_type_tf2pytorch.tar \
        --nr_inference_workers=2 \
        --nr_post_proc_workers=2 \
        wsi \
        --input_dir="$in_dir" \
        --output_dir="$hover_out_dir" \
        

#END

python generate_measurments.py "$@"
python generate_qupath.py "$@"

echo \done
