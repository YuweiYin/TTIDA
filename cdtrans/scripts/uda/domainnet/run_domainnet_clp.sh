model=$1
if [ ! -n "$1" ]
then 
    echo 'please input the model para: {deit_base, deit_small}'
    exit 8
fi
if [ $model == 'deit_base' ]
then
    model_type='uda_vit_base_patch16_224_TransReID'
    gpus="('0,1')"
else
    model='deit_small'
    model_type='uda_vit_small_patch16_224_TransReID'
    gpus="('0')"
fi
for target_dataset in 'painting' 'quickdraw' 'real' 'sketch' 'infograph'
do
    python train.py --config_file configs/uda.yml MODEL.DEVICE_ID $gpus \
    OUTPUT_DIR './log/uda/'$model'/domainnet/clipart2'$target_dataset \
    MODEL.PRETRAIN_PATH './log/pretrain/'$model'/domainnet/Clipart/transformer_10.pth' \
    DATASETS.ROOT_TRAIN_DIR './data/domainnetclipart.txt' \
    DATASETS.ROOT_TRAIN_DIR2 './data/domainnet/'$target_dataset'.txt' \
    DATASETS.ROOT_TEST_DIR './data/domainnet/'$target_dataset'.txt' \
    DATASETS.NAMES "DomainNet" DATASETS.NAMES2 "DomainNet" \
    MODEL.Transformer_TYPE $model_type
done
