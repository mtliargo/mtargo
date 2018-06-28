import os
import numpy as np
from os.path import join
from PIL import Image

data_dir = '/home/mli/Data'
in_file = join(data_dir, 'Exp/CARLA_gen18/e000045/Seg/00000006.png')
out_file = join(data_dir, 'Exp/CARLA_gen19/ego-vehicle.png')


img = np.array(Image.open(in_file))
img = img[:, :, 0]
mask = img == 10
assert mask.any()
mask = 255*mask.astype(np.uint8)
Image.fromarray(mask).save(out_file)
