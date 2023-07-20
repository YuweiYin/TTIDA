#!/bin/bash

CUDA=${1}
echo "CUDA: ${CUDA}"

for n_img_syn in "0" "20" "50" "100"; do
  data="office_31"
  #syn_type="base"
  syn_type="upsample"
  epochs=50
  SAVE_DIR="./log/${data}"
  mkdir -p "${SAVE_DIR}"
  for seed in "7" "17" "42"; do
    for source_domain in "amazon" "dslr" "webcam"; do
      for target_domain in "amazon" "dslr" "webcam"; do
        if [[ "${source_domain}" != "${target_domain}" ]]; then
          echo -e "\n\n *** CDTrans (pretrain) on ${data} [syn_type ${syn_type}] [seed ${seed}] (${source_domain} -> ${target_domain}) with ${n_img_syn} synthetic images per class"
          python train.py \
            --cuda "${CUDA}" --data "${data}" --syn_type "${syn_type}" --seed "${seed}" \
            --source_domain "${source_domain}" --target_domain "${target_domain}" \
            --n_img_syn ${n_img_syn} --epochs ${epochs} --pretrain \
            2>&1 | tee "${SAVE_DIR}/${data}_${source_domain}_to_${target_domain}_syn${n_img_syn}_pretrain_${seed}.log"
        fi
      done
    done
  done
done
