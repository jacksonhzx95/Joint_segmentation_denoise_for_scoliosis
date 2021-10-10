import argparse
import os.path as osp
from PIL import Image
import mmcv
#from cityscapesscripts.preparation.json2labelImg import json2labelImg

#
# def convert_json_to_label(json_file):
#     label_file = json_file.replace('_polygons.json', '_labelTrainIds.png')
#     json2labelImg(json_file, label_file, 'trainIds')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert scoliosis annotations to TrainIds')
    parser.add_argument('dataset_path',
                        default='/mnt/sd2/Semantic_Seg/mmsegmentation_for_3classes/data/scoliosis3classes',
                        help='dataset path')
    parser.add_argument('--gt-dir', default='groundtruth', type=str)
    parser.add_argument('-o', '--out-dir', default='/mnt/sd2/Semantic_Seg/mmsegmentation_for_3classes/data/scoliosis3classes',
                        help='output path')
    parser.add_argument(
        '--nproc', default=2, type=int, help='number of process')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    scoliosis_path = args.dataset_path
    out_dir = args.out_dir if args.out_dir else scoliosis_path
    mmcv.mkdir_or_exist(out_dir)

    gt_dir = osp.join(scoliosis_path, args.gt_dir)

    split_names = ['train', 'test']

    for split in split_names:
        filenames = []
        for poly in mmcv.scandir(
                osp.join(gt_dir, split), '_bone_3.png', recursive=True):
            with Image.open(osp.join(gt_dir, split, poly)) as im:
                im = im.point(lambda i: i / 255)
                r, g, b = im.split()
                g = g.point(lambda i: i * 2)
                mask_g = g.point(lambda i: i > 0 and 255)
                b = b.point(lambda i: i * 3)
                mask_b = b.point(lambda i: i > 0 and 255)
                r.paste(g, None, mask_g)
                r.paste(b, None, mask_b)
                r.save(osp.join(gt_dir, split, poly))
            filenames.append(poly.replace('_bone_3.png', ''))


        with open(osp.join(out_dir, f'{split}.txt'), 'w') as f:
            f.writelines(f + '\n' for f in filenames)

if __name__ == '__main__':
    main()
