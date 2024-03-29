#!/bin/bash

CUDA=${1}
#export CUDA_VISIBLE_DEVICES=${CUDA}
echo "CUDA: ${CUDA}"

SAVE_DIR="./log/img_clf_cifar100_adv"
mkdir -p "${SAVE_DIR}"

net="resnet101"
data="cifar100"
max_epoch="200"

function train_img_clf() {
  n_img_train=${1}
  n_img_syn=${2}
  syn_type=${3}
  epoch=${4}
  seed=${5}
  echo -e "\n\n\n >>> training adversarial ${net} on ${data} [syn_type ${syn_type}] [seed ${seed}]: ${n_img_train} + ${n_img_syn} per class"
  python run_train.py \
    --cuda "${CUDA}" --net "${net}" --data "${data}" --syn_type "${syn_type}" --seed "${seed}" --adversarial \
    --n_img_train "${n_img_train}" --n_img_syn "${n_img_syn}" --epoch "${epoch}" --b 128 \
    2>&1 | tee "${SAVE_DIR}/${data}_${net}_${syn_type}_seed${seed}_${n_img_train}_${n_img_syn}.log"
}

for seed in "7" "17" "42"; do
  train_img_clf "10" "0" "glide_upsample" "${max_epoch}" "${seed}"
  train_img_clf "10" "2" "glide_upsample" "${max_epoch}" "${seed}"
  train_img_clf "10" "5" "glide_upsample" "${max_epoch}" "${seed}"
  train_img_clf "10" "10" "glide_upsample" "${max_epoch}" "${seed}"

  train_img_clf "50" "0" "glide_upsample" "${max_epoch}" "${seed}"
  train_img_clf "50" "10" "glide_upsample" "${max_epoch}" "${seed}"
  train_img_clf "50" "25" "glide_upsample" "${max_epoch}" "${seed}"
  train_img_clf "50" "50" "glide_upsample" "${max_epoch}" "${seed}"

  train_img_clf "100" "0" "glide_upsample" "${max_epoch}" "${seed}"
  train_img_clf "100" "20" "glide_upsample" "${max_epoch}" "${seed}"
  train_img_clf "100" "50" "glide_upsample" "${max_epoch}" "${seed}"
  train_img_clf "100" "100" "glide_upsample" "${max_epoch}" "${seed}"

  train_img_clf "250" "0" "glide_upsample" "${max_epoch}" "${seed}"
  train_img_clf "250" "50" "glide_upsample" "${max_epoch}" "${seed}"
  train_img_clf "250" "125" "glide_upsample" "${max_epoch}" "${seed}"
  train_img_clf "250" "250" "glide_upsample" "${max_epoch}" "${seed}"

  train_img_clf "500" "0" "glide_upsample" "${max_epoch}" "${seed}"
  train_img_clf "500" "100" "glide_upsample" "${max_epoch}" "${seed}"
  train_img_clf "500" "250" "glide_upsample" "${max_epoch}" "${seed}"
  train_img_clf "500" "500" "glide_upsample" "${max_epoch}" "${seed}"
done
