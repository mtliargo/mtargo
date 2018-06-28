'''
Generate list of images
'''

import sys
sys.path.insert(0, '../util')
import platform_config as pc

import os, glob
from os.path import join
from PIL import Image
from shutil import copy2


data_dir = pc.data_dir
mkdir2 = pc.mkdir2

in_dir = join(data_dir, 'Exp/CARLA_gen18')
out_dir = mkdir2(join(data_dir, 'Exp/CARLA_gen18_gather1'))

out_size = (1024, 512)

list_name = join(in_dir, 'list-rgb-0.1.txt')
content = open(list_name).readlines()
file_list = [x.strip() for x in content]

copy2(list_name, out_dir)

for f in file_list:
    img = Image.open(join(in_dir, f))
    img = img.resize(out_size, Image.BICUBIC)
    mkdir2(join(out_dir, os.path.dirname(f)))
    img.save(join(out_dir, f))

