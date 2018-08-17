import sys; sys.path.insert(0, '../util')
from platform_config import data_dir, mkdir2

from os.path import join, basename
import numpy as np
from html4vision import Col, imagetable

import argparse, glob
from os.path import join

from datasets.cityscapes_info import labels

parser = argparse.ArgumentParser()
parser.add_argument('--exp-name', type=str, default='C20_S1_0.1_CS')
# parser.add_argument('--dir-struct', type=str, default='deeplab')
# parser.add_argument('--class-subset', type=int, nargs='+', default=None)

parser.add_argument('--dir-struct', type=str, default='mtpt')
parser.add_argument('--class-subset', type=int, nargs='+', default=[0, 1, 2, 3, 4, 5, 7, 8, 11, 13, 19])

opts = parser.parse_args()

exp_name = opts.exp_name
if opts.dir_struct == 'deeplab':
    exp_dir = join(data_dir, 'Exp/DeepLabCityscapes', exp_name)
    predict_dir = join(exp_dir, 'predict')
    out_dir = mkdir2(join(exp_dir, 'vis'))
    out_name = 'index.html'
    reorg_dir = join(out_dir, '../../../../Cityscapes/ReOrg')
    inputs = join(reorg_dir, 'Image-512-JPG/val/*.jpg')
    outputs = join(out_dir, '../predict/val/*.png')
    targets = join(reorg_dir, 'SegColor20-1024/val/*.png')
    targets_org = join(reorg_dir, 'SegColor35-1024/val/*.png')
    external_path = 'http://mt1080.pc.cs.cmu.edu:21368/Exp/DeepLabCityscapes/' + exp_name + '/vis/' + out_name
elif opts.dir_struct == 'mtpt':
    predict_dir = join(data_dir, 'Exp/Cityscapes/mtpt', 'output', exp_name)
    out_dir = predict_dir
    out_name = 'advanced.html'
    reorg_dir = join(out_dir, '../../../../../Cityscapes/ReOrg')
    inputs = join(reorg_dir, 'Image-512-JPG/val/*.jpg')
    outputs = join(out_dir, 'pred/*.png')
    targets = join(reorg_dir, 'SegColor20-1024/val/*.png')
    targets_org = join(reorg_dir, 'SegColor35-1024/val/*.png')
    external_path = 'http://mt1080.pc.cs.cmu.edu:21368/Exp/Cityscapes/mtpt/output/' + exp_name + '/'  + out_name
else:
    raise ValueError('Undefined directory structure')

def shorten(x):
    shape = x.shape
    s = np.array2string(100*x.flatten(), precision=1, suppress_small=True, threshold=x.size)[1:-1]
    x = np.fromstring(s, sep=' ')
    x = x.reshape(shape)
    return x

summary = shorten(np.loadtxt(join(predict_dir, 'eval-val-summary.txt')))
summary_per_class = shorten(np.loadtxt(join(predict_dir, 'eval-val-summary-per-class.txt')))
per_inst = shorten(np.loadtxt(join(predict_dir, 'eval-val-per-inst.txt')))
per_inst_per_class = shorten(np.loadtxt(join(predict_dir, 'eval-val-per-inst-per-class.txt')))

train_class_name = [label.category + '.' + label.name for label in labels if label.trainId != 255 and label.trainId >= 0]
if opts.class_subset is not None:
    train_class_name = [train_class_name[i] for i in opts.class_subset[:-1]]
sel = None # all
# sel = 6

overlay_col = Col('overlay', '', inputs, sel, 'opacity: 0.15')
cols = [
    Col('id1', 'ID'),
    Col('text', 'mIoU', per_inst[:, 0], sel),
    Col('img', 'Input Image', inputs, sel),
    Col('img', 'Output Segmentation', outputs, sel),
    overlay_col,
    Col('img', 'Ground Truth (standard, 20)', targets, sel),
    overlay_col,
    Col('img', 'Ground Truth (original, 35)', targets_org, sel),
    overlay_col,
    Col('text', 'mAccu', per_inst[:, 1], sel),
    Col('text', 'Accu', per_inst[:, 2], sel),
]

for i in range(len(train_class_name)):
    cols.append(Col('text', train_class_name[i], per_inst_per_class[:, i], sel))

summaryrow = [
    'S', summary[0], *(7*['']), summary[1], summary[2],
    *summary_per_class
]

assert(len(summaryrow) == len(cols))

out_path = join(out_dir, out_name)


imagetable(cols, out_path, exp_name,
    imsize=(512, 256), overlaytoggle=True,
    pathrep=out_dir + '/',
    stickyheader=True,
    sortcol=1, sortable=True,
    sortstyle='metro-dark', zebra=True,
    summaryrow=summaryrow,
    style='.html4vision tr.static td {background-color: #ffc044 !important}')

print('Generated at ' + out_path)
print('External path: ' + external_path)

