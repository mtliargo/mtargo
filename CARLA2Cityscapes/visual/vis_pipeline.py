## Setting up platform dependent stuffs
import sys; sys.path.insert(0, '../../util')
from platform_config import data_dir, mkdir2


from os.path import join, basename
from html4vision import Col, imagetable

out_dir = mkdir2(join(data_dir, 'Exp/C2C/Visual'))
exp_name = 'test5'
# exp_name = basename(__file__)[:-3]


seq_dir = join(data_dir, 'Exp/CARLA_gen19/e000001')
exp_dir = join(data_dir, 'Exp/C2C', exp_name)

seg_colors = join(seq_dir, 'SegColor/*.png')
source_imgs = join(seq_dir, 'RGB/*.png')

matched = join(exp_dir, 'SegMatch-256-Vis/*.jpg')
canvas = join(exp_dir, 'Canvas-512-Vis/*.jpg')
refined = join(exp_dir, 'Refined-512/*.png')

road1_source = join(exp_dir, 'SegMatch-256-VisEx/*_road1.jpg')
car1_source = join(exp_dir, 'SegMatch-256-VisEx/*_car1.jpg')
car2_source = join(exp_dir, 'SegMatch-256-VisEx/*_car2.jpg')
car3_source = join(exp_dir, 'SegMatch-256-VisEx/*_car3.jpg')
building1_source = join(exp_dir, 'SegMatch-256-VisEx/*_building1.jpg')

sel = None # all

overlay_col = Col('overlay', '', seg_colors, sel, 'opacity: 0.15')
cols = [
    Col('id1', 'ID'), # 1-based indexing
    Col('img', 'CARLA RGB', source_imgs, sel),
    overlay_col,
    Col('img', 'Input Label', seg_colors, sel),
    Col('img', 'Output Image', refined, sel),
    overlay_col,
    Col('img', 'Segment Matching', matched, sel),
    overlay_col,
    Col('img', 'Canvas', canvas, sel),
    overlay_col,
    Col('img', 'Road 1 Source', road1_source, sel),
    Col('img', 'Car 1 Source', car1_source, sel),
    Col('img', 'Car 2 Source', car2_source, sel),
    Col('img', 'Car 3 Source', car3_source, sel),
    Col('img', 'Building 1 Source', building1_source, sel),    
]

out_path = join('', exp_name + '.html')

imagetable(cols, out_path, exp_name, imsize=(512, 256), interactive=True)


