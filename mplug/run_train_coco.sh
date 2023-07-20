#!/bin/bash

CUDA=${1}
echo "CUDA: ${CUDA}"

SAVE_DIR="./log"
mkdir -p "${SAVE_DIR}"

run_caption_mplug(){
  n_img=${1}
  n_img_syn=${2}
  seed=${3}
  echo -e "\n\n\n >>> training mplug on MS COCO Image Captioning [seed ${seed}]: ${n_img} + syn data ${n_img_syn}"
  python caption_mplug.py \
    --cuda "${CUDA}" --seed "${seed}" \
    --n_img "${n_img}" --n_img_syn "${n_img_syn}" \
    --bsz_train 32 --bsz_test 32 \
    2>&1 | tee "${SAVE_DIR}/run_coco_captioning_mplug_seed${seed}_${n_img}_syn${n_img_syn}.log"
}

for seed in "7" "17" "42"; do
  run_caption_mplug 5000 0 "${seed}"
  run_caption_mplug 5000 1000 "${seed}"
  run_caption_mplug 5000 5000 "${seed}"
  run_caption_mplug 5000 10000 "${seed}"
  run_caption_mplug 5000 50000 "${seed}"

  run_caption_mplug 10000 0 "${seed}"
  run_caption_mplug 10000 5000 "${seed}"
  run_caption_mplug 10000 10000 "${seed}"
  run_caption_mplug 10000 50000 "${seed}"
  run_caption_mplug 10000 100000 "${seed}"

  run_caption_mplug 50000 0 "${seed}"
  run_caption_mplug 50000 10000 "${seed}"
  run_caption_mplug 50000 50000 "${seed}"
  run_caption_mplug 50000 100000 "${seed}"

  run_caption_mplug 100000 0 "${seed}"
  run_caption_mplug 100000 10000 "${seed}"
  run_caption_mplug 100000 50000 "${seed}"
  run_caption_mplug 100000 100000 "${seed}"

  run_caption_mplug 200000 0 "${seed}"
  run_caption_mplug 200000 50000 "${seed}"
  run_caption_mplug 200000 100000 "${seed}"
  run_caption_mplug 200000 200000 "${seed}"
done
