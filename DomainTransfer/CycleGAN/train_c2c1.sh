set -ex
python train.py \
    --display_port 31031 \
    --dataset_mode unaligned_list \
    --datarootA ../../Data/Exp/CARLA_gen18_gather1 \
    --listA ../../Data/Exp/CARLA_gen18_gather1/list-rgb-0.1.txt \
    --datarootB ../../Data/Cityscapes/ReOrg/Image-512/train \
    --listB ../../Data/Cityscapes/ReOrg/train-image.txt \
    --name c2c1 \
    --model cycle_gan \
    --pool_size 50 \
    --no_dropout \
    --batchSize 4 \
    --resize_or_crop scale_width \
    --fineSize 256 \
    # --resize_or_crop none \
