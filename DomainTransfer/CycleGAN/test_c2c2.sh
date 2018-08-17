set -ex
python test.py \
	--how_many=100000 \
	--phase test \
	--no_dropout \
	--model cycle_gan \
	--name c2c2 \
	--display_port 31031 \
	--dataset_mode unaligned_list \
	--resize_or_crop scale_width \
    --fineSize 256 \
    --datarootA ../../Data/CarlaGen/C18_W2_S1/RGB-1024/test \
    --listA ../../Data/CarlaGen/C18_W2_S1/test.txt \
    --datarootB ../../Data/Cityscapes/ReOrg/Image-512/train \
    --listB ../../Data/test.txt \
	
	
	
	
#	--dataroot ./datasets/maps --name maps_cyclegan  
