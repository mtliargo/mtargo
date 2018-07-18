import sys; sys.path.insert(0, '../util')
from platform_config import data_dir

import os
import numpy as np
from os.path import join
from PIL import Image

in_file = join(data_dir, 'Exp/CARLA_vid/e000001/Seg/00000300.png')
out_file = join(data_dir, 'Exp/CARLA_vid/ego-vehicle.png')

# in_file = join(data_dir, 'Exp/CARLA_gen19/e000015/Seg/00000001.png')
# out_file = join(data_dir, 'Exp/CARLA_gen19/ego-vehicle.png')


img = np.array(Image.open(in_file))
img = img[:, :, 0]
mask = img == 10
assert mask.any()
mask = 255*mask.astype(np.uint8)
Image.fromarray(mask).save(out_file)
