import sys; sys.path.insert(0, '../util')
from platform_config import data_dir, mkdir2

import argparse, glob
from os.path import join, basename
from PIL import Image
import numpy as np
from sklearn.metrics import confusion_matrix

from datasets.cityscapes_info import labels

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, default='C20_S1_0.1_CS')
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--save-each-confmat', type=bool, default=True)
    # parser.add_argument('--dir-struct', type=str, default='deeplab')
    # parser.add_argument('--class-subset', type=int, nargs='+', default=None)

    parser.add_argument('--dir-struct', type=str, default='mtpt')
    parser.add_argument('--class-subset', type=int, nargs='+', default=[0, 1, 2, 3, 4, 5, 7, 8, 11, 13, 19])
    # parser.add_argument('--predict-list', type=str,
    #                     help='list of predictions')
    # parser.add_argument('--target-list', type=str,
    #                     help='list of targets')
    # parser.add_argument('--output-dir', type=str,
    #                     help='directory stores the output')
    
    ##
    opts = parser.parse_args()

    opts.target_list = sorted(glob.glob(join(data_dir, 'CityScapes/ReOrg/SegColor20-1024/val/*.png')))
    if opts.dir_struct == 'deeplab':
        predict_dir = join(data_dir, 'Exp/DeepLabCityscapes', opts.exp_name, 'predict')
        opts.predict_list = sorted(glob.glob(join(predict_dir, opts.split, '*.png')))
        record_dir = mkdir2(join(data_dir, 'Exp/DeepLabCityscapes/record'))
    elif opts.dir_struct == 'mtpt':
        predict_dir = join(data_dir, 'Exp/Cityscapes/mtpt/output', opts.exp_name)
        opts.predict_list = sorted(glob.glob(join(predict_dir, 'pred', '*.png')))
        record_dir = mkdir2(join(data_dir, 'Exp/Cityscapes/record'))
    else:
        raise ValueError('Undefined directory structure')
    opts.outfile_record_entry = join(record_dir, opts.exp_name + '.csv')
    opts.outfile_summary = join(predict_dir, 'eval-' + opts.split + '-summary.txt')
    opts.outfile_per_class = join(predict_dir, 'eval-' + opts.split + '-summary-per-class.txt')
    opts.outfile_per_inst = join(predict_dir, 'eval-' + opts.split + '-per-inst.txt')
    opts.outfile_per_inst_per_class = join(predict_dir, 'eval-' + opts.split + '-per-inst-per-class.txt')
    if opts.save_each_confmat:
        opts.confmat_dir = mkdir2(join(predict_dir, 'confmat', opts.split))
        opts.confmat_total = join(predict_dir, 'confmat', opts.split + '.txt')

    return opts

def parse_confmat(cmat):
    # the metrics are only computed over class that appear in the target or the prediction
    # assume cmat is not all zeros
    n_class = cmat.shape[0]
    sum_over_row = cmat.sum(1)
    sum_over_col = cmat.sum(0)
    cmat_diag = cmat.diagonal()
    union = sum_over_row + sum_over_col - cmat_diag

    accu = cmat_diag.sum() / sum_over_row.sum()

    # standard
    # class_absent = union == 0

    # gt-only
    class_absent = sum_over_row == 0

    union[class_absent] = 1
    iou_class = cmat_diag/union
    m_iou = iou_class.sum()/(n_class - class_absent.sum())

    class_absent = sum_over_row == 0
    sum_over_row[class_absent] = 1
    accu_class = cmat_diag/sum_over_row
    m_accu = accu_class.sum()/(n_class - class_absent.sum())

    summary = [m_iou, m_accu, accu]
    per_class = iou_class
    return summary, per_class


def main():
    opts = parse_args()

    n = len(opts.target_list)
    n_class = 20

    per_inst = n*[None]
    per_inst_per_class = n*[None]
    # [mIoU, mAccu, IoU]
    cmat_class = 0

    # the last class (void, unlabeled, others) is ignored
    if opts.class_subset is not None:
        class_mapping = np.full(n_class, len(opts.class_subset)-1, dtype=np.int64)
        for i, c in enumerate(opts.class_subset):
            class_mapping[c] = i
        n_class = len(opts.class_subset)

    label_set = np.arange(n_class-1)

    for i in range(n):
        print('Processing %d/%d - %s' % (i+1, n, opts.predict_list[i]))
        target = Image.open(opts.target_list[i])
        predict = Image.open(opts.predict_list[i])
        assert len(predict.size) == 2
        if predict.size != target.size:
            predict = predict.resize(target.size, Image.NEAREST)
        target = np.array(target).flatten()
        predict = np.array(predict).flatten()
        if opts.class_subset is not None:
            target = class_mapping[target]
        # assert(np.all(target < n_class))
        # assert(np.all(predict < n_class))

        cmat_inst = confusion_matrix(target, predict, label_set)
        if opts.save_each_confmat:
            np.savetxt(join(opts.confmat_dir, basename(opts.predict_list[i])[:-3] + 'txt'), cmat_inst, '%d')

        per_inst[i], per_inst_per_class[i] = parse_confmat(cmat_inst)
        cmat_class += cmat_inst

        print(per_inst[i])
    
    if opts.save_each_confmat:
        np.savetxt(opts.confmat_total, cmat_class, '%d')

    summary, summary_per_class = parse_confmat(cmat_class)

    per_inst = np.array(per_inst).squeeze()
    per_inst_per_class = np.array(per_inst_per_class).squeeze()

    print('Summary:')
    print(summary)
    print('Per-class mIoU:')
    print('mIoU (min)')
    print('%.2g %.2g' % (100*summary[0], 100*per_inst[:, 0].min()))

    n_class_per_row = 4
    train_class_name = [label.name for label in labels if label.trainId != 255 and label.trainId >= 0]
    if opts.class_subset is not None:
        train_class_name = [train_class_name[i] for i in opts.class_subset[:-1]]
    for i in range(len(train_class_name)):
        print('%s: %.1f' % (train_class_name[i], 100*summary_per_class[i]), end='')
        if (i + 1) % n_class_per_row:
            print('\t\t', end='')
        else:
            print('')    

    nonzeromin = lambda x: x[x>0].min() # wrong!

    if opts.class_subset is not None:
        with open(opts.outfile_record_entry, 'w') as f:
            f.write('%s,%.1f,%.1f\n' % (
                        opts.exp_name,
                        100*summary[0],
                        100*per_inst[:, 0].min(),                                       
                    ))
    else:
        with open(opts.outfile_record_entry, 'w') as f:
            f.write('%s,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f\n' % (
                        opts.exp_name,
                        100*summary[0],
                        100*per_inst[:, 0].min(),
                        100*summary_per_class[11],
                        100*nonzeromin(per_inst_per_class[:, 11]),
                        100*summary_per_class[12],
                        100*nonzeromin(per_inst_per_class[:, 12]),                                        
                    ))
    save_fmt = '%.18g'
    np.savetxt(opts.outfile_summary, summary, save_fmt)
    np.savetxt(opts.outfile_per_class, summary_per_class, save_fmt)
    np.savetxt(opts.outfile_per_inst, per_inst, save_fmt)
    np.savetxt(opts.outfile_per_inst_per_class, per_inst_per_class, save_fmt)

if __name__ == '__main__':
    main()
        