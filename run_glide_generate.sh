#!/bin/bash

CUDA=${1}
echo "CUDA: ${CUDA}"

function glide_generate() {
  syn_data=${1}
  n_img=${2}
  seed=${3}
  echo -e "\n\n\n >>> Generate Synthetic Images by GLIDE: ${n_img} per prompt on ${syn_data} [seed ${seed}]"
  python run_glide_generate.py \
    --cuda "${CUDA}" --syn_data "${syn_data}" --n_img "${n_img}" --seed "${seed}" --verbose
}

glide_generate "cifar100" 500 "42"
glide_generate "office_31" 100 "42"
glide_generate "office_home" 100 "42"
glide_generate "imageclef_da" 100 "42"
glide_generate "coco_cap" 10 "42"
glide_generate "coco_cap_sent" 10 "42"
glide_generate "coco_cap_ner" 10 "42"
