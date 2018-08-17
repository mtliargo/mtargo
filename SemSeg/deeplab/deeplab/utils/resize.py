from os.path import join, isdir, basename
from os import makedirs
from glob import glob
from PIL import Image

data_dir = 'Data'
in_dir = join(data_dir, 'CarlaGen/C20_S1/RGB-1024')
out_dir = join(data_dir, 'CarlaGen/C20_S1/RGB-512')

splits = ['train', 'val', 'test']

for s in splits:
    in_dir_split = join(in_dir, s)
    out_dir_split = join(out_dir, s)
    if not isdir(out_dir_split):
        makedirs(out_dir_split)
        
    files = glob(join(in_dir_split, '*.png'))

    for i, f in enumerate(files):
        print('Processing %d/%d - %s' % (i, len(files), f))
        img = Image.open(f)
        img = img.resize((1024, 512), Image.BICUBIC)
        img.save(join(out_dir_split, basename(f)))
    