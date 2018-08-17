import sys; sys.path.insert(0, '../util')
from platform_config import data_dir, mkdir2

from glob import glob
from os.path import join
from shutil import copy2
from PIL import Image

split = 'train'
in_dir = join(r'D:\c2c2', split, 'images')
out_dir128 = mkdir2(join(data_dir, 'CarlaGen/C18_W2_S1/CycleGAN-128', split))
out_dir1024 = mkdir2(join(data_dir, 'CarlaGen/C18_W2_S1/CycleGAN-1024', split))


file_list = sorted(glob(join(in_dir, '*_fake_B.png')))
n = len(file_list)

for i, f in enumerate(file_list):
    print('Processing %d/%d - %s' % (i, n, f))
    copy2(f, join(out_dir128, '%08d.png' % (i+1)))
    img = Image.open(f)
    img = img.resize((2048, 1024), Image.BICUBIC)
    img.save(join(out_dir1024, '%08d.png' % (i+1)))

