#!/bin/bash

CUDA=${1}
echo "CUDA: ${CUDA}"

function gpt_generate() {
  syn_data=${1}
  n_sent=${2}
  seed=${3}
  echo -e "\n\n\n >>> Generate Descriptions by GPT-2: ${n_sent} per label text on ${syn_data} [seed ${seed}]"
  python run_gpt_generate.py \
    --cuda "${CUDA}" --syn_data "${syn_data}" --n_sent "${n_sent}" --seed "${seed}" --verbose
}

gpt_generate "cifar100" 500 "42"
gpt_generate "office_31" 100 "42"
gpt_generate "office_home" 100 "42"
gpt_generate "coco_cap" 1 "42"
gpt_generate "coco_cap_sent" 10 "42"
gpt_generate "coco_cap_ner" 10 "42"
