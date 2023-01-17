#!/bin/sh

in_dir=../data_in
out_dir=../out

python extract_tiles.py

for f in "$in_dir"/*; do
    if [ -f "$f" ]; then
        filename=$(basename -- "$f")
        extension="${filename##*.}"
        filename="${filename%.*}"
        
        tile_dir="$out_dir"/tiles/"$filename"
        hover_out_dir="$out_dir"/hover/"$filename"
        
        python hover_net/run_infer.py \
            --gpu=0 \
            --nr_types=6 \
            --type_info_path=hover_net/type_info.json \
            --batch_size=16 \
            --model_mode=fast \
            --model_path=../checkpoints/hovernet_fast_pannuke_type_tf2pytorch.tar \
            --nr_inference_workers=2 \
            --nr_post_proc_workers=2
            \
            tile \
            --input_dir="$tile_dir" \
            --output_dir="$hover_out_dir" \
            --mem_usage=0.1
    fi
done

python generate_masks.py
python blurr_masks.py
python generate_qupath.py

echo \done