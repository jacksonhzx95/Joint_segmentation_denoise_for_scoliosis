import argparse
import mmcv
import torch
from mmcv.parallel import MMDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmseg.apis import single_gpu_test_joint
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor


def parse_args():
    parser = argparse.ArgumentParser(
        description='mmseg test (and eval) a model')
    parser.add_argument('config',
                        default='./configs/scoliosis_joint_learning/joint_learning.py',
                        help='test config file path')
    parser.add_argument('checkpoint',
                        default='./checkpoints/fpn101_aug.pth',
                        help='checkpoint file')
    parser.add_argument('--out',
                        default=None,
                        help='output result file in pickle format')

    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--show_denoise', action='store_true', help='show results')
    parser.add_argument(
        '--show_dir', default='./scoliosis_inference',
        help='directory where painted images will be saved')
    parser.add_argument(
        '--show_mask', default='./scoliosis_inference_mask',
        help='directory where painted images will be saved')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
           or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = mmcv.Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    cfg.data.test.test_mode = False

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

    # build the model and load checkpoint
    model = build_segmentor(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    load_checkpoint(model, args.checkpoint, map_location='cpu')

    CLASSES = ('Background', 'Rib', 'Thoracic', 'Lumbar')

    PALETTE = [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]]
    model.CLASSES = CLASSES
    model.PALETTE = PALETTE

    model = MMDataParallel(model, device_ids=[0])
    outputs, _ = single_gpu_test_joint(model, data_loader, show=args.show,
                                     out_dir=args.show_dir, show_gan=args.show_denoise,
                                     out_mask=args.show_mask)




if __name__ == '__main__':
    main()
