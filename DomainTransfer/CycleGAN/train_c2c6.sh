set -ex
python train.py \
    --display_port 31031 \
    --dataset_mode unaligned_list \
    --datarootA ../../Data/CarlaGen/C20_S1/RGB-512/train \
    --listA ../../Data/CarlaGen/C20_S1/train-0.1.txt \
    --datarootB ../../Data/Cityscapes/ReOrg/Image-512/train \
    --listB ../../Data/Cityscapes/ReOrg/train-image.txt \
    --name c2c6 \
    --model cycle_gan \
    --pool_size 50 \
    --no_dropout \
    --batchSize 4 \
    --resize_or_crop scale_width \
    --fineSize 256 \
    # --resize_or_crop none \
