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
for target_dataset in 'Clipart' 'Art' 'Product'
do
    python train.py --config_file configs/uda.yml MODEL.DEVICE_ID $gpus \
    OUTPUT_DIR './log/uda/'$model'/office-home/RealWorld2'$target_dataset \
    MODEL.PRETRAIN_PATH './log/pretrain/'$model'/office-home/Real_World/transformer_10.pth' \
    DATASETS.ROOT_TRAIN_DIR './data/OfficeHomeDataset/Real_World.txt' \
    DATASETS.ROOT_TRAIN_DIR2 './data/OfficeHomeDataset/'$target_dataset'.txt' \
    DATASETS.ROOT_TEST_DIR './data/OfficeHomeDataset/'$target_dataset'.txt' \
    DATASETS.NAMES "OfficeHome" DATASETS.NAMES2 "OfficeHome" \
    MODEL.Transformer_TYPE $model_type
done
