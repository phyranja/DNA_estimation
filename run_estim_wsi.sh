#!/bin/sh

in_dir=../data_in
out_dir=../out

hover_out_dir="$out_dir"/hover/

# : <<'END'
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
        

# END

python generate_qupath.py -c estim_config.conf --use_tiles False

echo \done
