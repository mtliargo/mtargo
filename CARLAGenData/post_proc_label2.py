'''
Deal with CARLA labeling issues and colorize the segments
http://carla.readthedocs.io/en/latest/cameras_and_sensors/#camera-semantic-segmentation

'''

import sys; sys.path.insert(0, '../util')
from platform_config import data_dir, mkdir2

import os, glob
import numpy as np
from os.path import join
from PIL import Image
from skimage import measure
import cv2
# import matplotlib.pyplot as plt
from cityscapes import c2trainid, cm_train

# for converting to indexed images
cm = list(cm_train.flatten())

def proc_sky(img):
    # none_mask = img == 0
    # cc, n_cc = measure.label(none_mask, connectivity=2, return_num=True)
    # max_cc_size = 0
    # max_cc = None
    # for i in range(n_cc):
    #     this_cc = cc == (i+1)
    #     cc_size = this_cc.sum()
    #     if cc_size > max_cc_size:
    #         max_cc = this_cc
    #         max_cc_size = cc_size
    # img[max_cc] = 14

    # h, w = img.shape
    # assert h & 2 == 0
    # none_mask = img == 0
    # top_mask = np.vstack( (np.ones((h//2, w), bool), np.zeros((h//2, w), bool)) )
    # img[none_mask & top_mask] = 14

    none_mask = img == 0
    cc = measure.label(none_mask, connectivity=2)
    top_cc = cc[0, :]
    top_cc_idx = np.unique(top_cc[top_cc != 0])
    sky_mask = np.isin(cc, top_cc_idx)
    img[sky_mask] = 14

    return img

def proc_car(img):
    h, w = img.shape
    car_mask = (img == 10).astype(np.uint8)
     # This point is always the ego-vehicle thus not other car masks
    seed = (w//2, h-1)
    dummy_mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(car_mask, dummy_mask, seed, 255)
    new_mask = car_mask != 255
    img[new_mask] = 10
    return img

single_mode = 0

if single_mode:
    ## single mode
    in_dir = join(data_dir, 'Exp/CARLA_gen20_town2/e000003/SegRaw')
    out_dir = mkdir2(join(data_dir, 'Exp/CARLA_gen20_town2/e000003/Seg'))
    mask_file = join(data_dir, 'Exp/CARLA_gen20_town2/ego-vehicle.png')

    mask = np.array(Image.open(mask_file), dtype=bool)

    files = glob.glob(join(in_dir, '*.png'))
    for f in files:
        img = np.array(Image.open(f))
        img = img[:, :, 0]
        img[mask] = 13
        img = proc_sky(img)
        img = proc_car(img) 
        img = Image.fromarray(c2trainid(img).astype(np.uint8))
        img.putpalette(cm)
        img.save(join(out_dir, os.path.basename(f)))
else:
## batch mode
    mask_file = join(data_dir, 'Exp/CARLA_gen20_town2/ego-vehicle.png')
    mask = np.array(Image.open(mask_file), dtype=bool)

    seqs = sorted(glob.glob(join(data_dir, 'Exp/CARLA_gen20_town2/e*')))
    seqs = list(filter(lambda s: os.path.isdir(s), seqs))

    for s in seqs:
        in_dir = join(s, 'SegRaw')
        out_dir = join(s, 'Seg')
        
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

        files = glob.glob(join(in_dir, '*.png'))
        for i, f in enumerate(files):
            print('Processing %d/%d - %s' % (i, len(files), f))
            img = np.array(Image.open(f))
            img = img[:, :, 0]
            img[mask] = 13
            img = proc_sky(img)
            img = proc_car(img)

            img = Image.fromarray(c2trainid(img).astype(np.uint8))
            img.putpalette(cm)
            img.save(join(out_dir, os.path.basename(f)))
