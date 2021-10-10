import argparse
import copy
import os
import os.path as osp
import time
import mmcv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from mmcv.runner import init_dist, build_optimizer
from mmcv.utils import Config, DictAction, get_git_hash
from mmseg import __version__
from mmseg.apis import single_gpu_test_joint
from mmseg.datasets import build_dataset, build_dataloader
from mmseg.models import build_segmentor
from mmseg.utils import collect_env, get_root_logger
from mmseg.models import builder
import sys
import torch.optim as optim

sys.setrecursionlimit(2000)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('--config', default='./configs/scoliosis_joint_learning/joint_learning.py', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--load-from',
        # default='/mnt/sd2/Semantic_Seg/mmsegmentation_for_3classes/work_dirs/fpn101_aug/iter_160000.pth',
        help='the checkpoint file to load weights from'
    )
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
             '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
             '(only applicable to non-distributed training)')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    if args.resume_from is not None:
        cfg.resume_from = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # create generators results file:
    gan_out_file = osp.join(cfg.work_dir, 'gen_results')
    gan_out_cln = osp.join(cfg.work_dir, 'cln')
    gan_out_train_file = osp.join(cfg.work_dir, 'gen_train_results')
    mmcv.mkdir_or_exist(osp.abspath(gan_out_file))
    mmcv.mkdir_or_exist(osp.abspath(gan_out_cln))
    mmcv.mkdir_or_exist(osp.abspath(gan_out_train_file))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%m%d_%H%M', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')
    meta['exp_name'] = osp.basename(args.config)

    model = build_segmentor(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    dis = builder.build_component(cfg.discriminator)
    if args.load_from is not None:
        cfg.load_from = args.load_from
        checkpoint = torch.load(args.load_from)
        model.load_state_dict(checkpoint)
    model = model.cuda()
    dis = dis.cuda()

    logger.info(model)
    datasets = build_dataset(cfg.data.train)
    dataset_gan = build_dataset(cfg.data.train_gan)

    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmseg version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmseg_version=f'{__version__}+{get_git_hash()[:7]}',
            config=cfg.pretty_text, )
    # add an attribute for visualization convenience
    model.CLASSES = dataset_gan.CLASSES
    data_loader = build_dataloader(
        datasets,
        cfg.data.samples_per_gpu,
        cfg.data.workers_per_gpu,
        len(cfg.gpu_ids),
        dist=distributed,
        seed=cfg.seed,
        drop_last=True)
    data_seg = iter(data_loader)
    data_loader_gan = DataLoader(
        dataset_gan,
        num_workers=cfg.data.workers_per_gpu,
        batch_size=8,
        shuffle=True,
        drop_last=True)
    data_gan = iter(data_loader_gan)

    if args.load_from is not None:
        cfg.load_from = args.load_from
        checkpoint = torch.load(args.load_from)
        model.load_state_dict(checkpoint)

    for params in model.parameters():
        params.requires_grad = False

    model.train(False)
    seg_params = []

    for params in model.backbone.parameters():
        params.requires_grad = True
        seg_params += [params]

    for params in model.neck.parameters():
        params.requires_grad = True
        seg_params += [params]
    for params in model.decode_head.parameters():
        params.requires_grad = True
        seg_params += [params]
    for params in model.feature_selection.parameters():
        params.requires_grad = True
        seg_params += [params]

    # freeze seg network

    # for params in model.parameters():
    #     params.requires_grad = False
    gen_params = []
    for params in model.backbone_gan.parameters():
        params.requires_grad = True
        gen_params += [params]
    for params in model.G_head.parameters():
        params.requires_grad = True
        gen_params += [params]
    for params in model.feature_selection.parameters():
        params.requires_grad = True
        gen_params += [params]

    optimizer_s = optim.SGD(seg_params, lr=0.001, momentum=0.9, weight_decay=0.0005)
    optimizer_g = optim.Adam(gen_params, lr=1e-4, weight_decay=0.0005, betas=(0.9, 0.999), amsgrad=True)
    optimizer_d = optim.Adam(dis.parameters(), lr=1e-4, weight_decay=0.0005, betas=(0.9, 0.999), amsgrad=True)
    scheduler_s = optim.lr_scheduler.CosineAnnealingLR(optimizer_s, T_max=Iter_Max, eta_min=1e-4)
    scheduler_g = optim.lr_scheduler.CosineAnnealingLR(optimizer_g, T_max=Iter_Max, eta_min=1e-6)
    scheduler_d = optim.lr_scheduler.CosineAnnealingLR(optimizer_d, T_max=Iter_Max, eta_min=1e-6)

    val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
    val_dataloader = build_dataloader(
        val_dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # loss
    loss_fc = nn.MSELoss(reduction='sum')
    dis_loss = nn.BCEWithLogitsLoss(reduction='sum')

    Iter_Max = 80000
    # start training
    for i in range(0, Iter_Max):
        scheduler_s.step()
        scheduler_g.step()
        scheduler_d.step()
        if i % 3 == 1:
            dis.train(True)
            model.train(False)
            dict_tar, dict_inp = next(data_gan, [None, None])
            if dict_inp == None:
                data_gan = iter(data_loader_gan)
                # print('next epoch')
                dict_tar, dict_inp = next(data_gan, [None, None])

            tensor_tar = Variable(copy.deepcopy(dict_tar['img']).cuda(),
                                  requires_grad=False)

            tensor_inp = Variable(copy.deepcopy(dict_inp['img'].cuda()),
                                  requires_grad=False)

            tensor_inp_seg_gt = torch.zeros(size=tensor_tar.size(),
                                            dtype=torch.long).cuda()
            valid = Variable(torch.ones(size=(tensor_inp.size(0), 1)).cuda(),
                             requires_grad=False)
            fake = Variable(torch.zeros(size=(tensor_inp.size(0), 1)).cuda(),
                            requires_grad=False)

            for i_dis in range(0, 3):
                '''first 3 iterations for training '''

                with torch.no_grad():
                    _, tensor_pre = model(img=tensor_inp,
                                          gt_semantic_seg=tensor_inp_seg_gt,
                                          img_metas=None)

                optimizer_d.zero_grad()
                validity_real = dis(tensor_tar)
                validity_fake = dis(tensor_pre)
                error_real = dis_loss(validity_real, valid)
                error_fake = dis_loss(validity_fake, fake)
                error = (error_real + error_fake) / 2
                error.backward()
                optimizer_d.step()

            # train gen
            model.train(True)
            dis.train(False)
            optimizer_g.zero_grad()
            _, out = model(img=tensor_inp,
                           gt_semantic_seg=tensor_inp_seg_gt,
                           img_metas=None)
            test_fake = dis(out)

            # dual adversarial learning
            test_real = dis(tensor_tar - out + tensor_inp)

            loss_den = loss_fc(out, tensor_inp)
            loss_gan = dis_loss(test_fake, valid) + 0.5 * dis_loss(test_real, fake)

            Loss = (0.0001 * loss_den + loss_gan) / tensor_tar.size(0)
            Loss.backward()
            optimizer_g.step()

            if i % 50 == 49:
                logger.info('#Iter: %4d, Rec loss: %.4f; GAN loss: %.4f' %
                            (i + 1, loss_den / tensor_tar.size(0),
                             loss_gan / tensor_tar.size(0)))

        # scheduler.step()
        data_batch = next(data_seg, None)
        if data_batch is None:
            data_seg = iter(data_loader)
            data_batch = next(data_seg, None)
        model.train()
        dis.eval()
        img_tensor = data_batch['img'].data[0]
        seg_tensor = data_batch['gt_semantic_seg'].data[0]
        in_metas = data_batch['img_metas'].data[0]
        img_tensor = img_tensor.cuda()
        seg_tensor = seg_tensor.cuda()
        optimizer_s.zero_grad()
        losses, _ = model(img=img_tensor, gt_semantic_seg=seg_tensor, img_metas=in_metas)
        loss, log_vars = model.parse_loss(losses)
        loss.backward()
        optimizer_s.step()
        if i % 50 == 49:
            logger.info('#Iter: %4d; decode.loss_seg: %.2f; decode.acc_seg: %.2f; loss: %.2f' %
                        (i + 1, log_vars['decode.loss_seg'],
                         log_vars['decode.acc_seg'],
                         log_vars['loss']))

        if i % 1000 == 999:
            torch.save(model.state_dict(),
                       os.path.join(cfg.work_dir, 'net_iter_' + str(i + 1) + '.pth'))
            model.train(False)
            model.eval()
            results, gen_results = single_gpu_test_joint(model,
                                                       val_dataloader, show=False, out_dir=gan_out_file, show_gan=True)
            eval_res = val_dataloader.dataset.evaluate(
                results, logger=logger)
            logger.info(eval_res)

        if i % 10000 == 9999:
            torch.save(dis.state_dict(),
                       os.path.join(cfg.work_dir, 'dis_iter_' + str(i + 1) + '.pth'))


if __name__ == '__main__':
    main()
