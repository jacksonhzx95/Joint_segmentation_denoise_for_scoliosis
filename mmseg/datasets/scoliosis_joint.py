import os.path as osp
import mmcv
from .builder import DATASETS
from .custom import CustomDataset
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import numpy as np
from .pipelines import Compose
import copy
import random


@DATASETS.register_module()
class ScoliosisDataset_joint(Dataset):
    """Scoliosis dataset.

    Args:
        split (str): Split txt file for Pascal VOC.
    """

    CLASSES = ('Background', 'Rib', 'Thoracic', 'Lumbar')

    PALETTE = [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]]

    def __init__(self,
                 pipeline,
                 img_dir,
                 img_suffix='.png',
                 cln_dir='train_cln',
                 cor_dir='train_cor',
                 split=None,
                 data_root=None,
                 test_mode=False,
                 ignore_index=255,
                 reduce_zero_label=False,
                 **kwargs):
        self.img_dir = img_dir
        self.img_suffix = img_suffix
        self.split = split
        self.data_root = data_root
        self.test_mode = test_mode
        self.ignore_index = ignore_index
        self.reduce_zero_label = reduce_zero_label
        self.pipeline = Compose(pipeline)
        self.cln_dir = cln_dir
        self.cor_dir = cor_dir
        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.img_dir):
                self.img_dir = osp.join(self.data_root, self.img_dir)
        if self.cln_dir is not None:
            if not osp.isabs(self.cln_dir):
                self.cln_dir = osp.join(self.img_dir, self.cln_dir)
        if self.cor_dir is not None:
            if not osp.isabs(self.cor_dir):
                self.cor_dir = osp.join(self.img_dir, self.cor_dir)

        # LOAD DATA Jackson_HUANG data: dict('cln': [], 'cor': [])
        self.cln_infos, self.nos_infos = self.load_data(self.cln_dir,
                                                        self.cor_dir,
                                                        self.img_suffix)
        # self.cln_patches, self.nos_patches = self.pre_processing(cln_level=1.2, nos_level=1.8)

    def __len__(self):
        return min(len(self.cln_infos), len(self.nos_infos))

    def __getitem__(self, item):
        if self.test_mode:
            nos = self.nos_infos[item]
            # cln_out = None
            nos_out = dict(img=nos['img'])
            # results = dict(img_info=img_info)
            # self.pre_pipeline(cln_patch)
            return self.pipeline(nos_out), nos_out
            # return
        else:
            cln_patch = dict(img_info=self.cln_infos[random.randint(0, 11111)])
            nos_patch = dict(img_info=self.nos_infos[item])

            cln_dict = self.pipeline(cln_patch)
            nos_dict = self.pipeline(nos_patch)
            cln_dict_out = dict(img=cln_dict['img'])
            nos_dict_out = dict(img=nos_dict['img'])
            # results = dict(img_info=img_info)
            # self.pre_pipeline(cln_patch)
            return cln_dict_out, nos_dict_out
        # return None

    def load_data(self, cln_dir, cor_dir,
                  img_suffix):
        cln_infos = []
        nos_infos = []

        for img in mmcv.scandir(cln_dir, img_suffix, recursive=True):
            img_file = osp.join(cln_dir, img)
            img_info = dict(filename=img_file)
            cln_infos.append(img_info)
        for img in mmcv.scandir(cor_dir, img_suffix, recursive=True):
            img_file = osp.join(cor_dir, img)
            img_info = dict(filename=img_file)
            nos_infos.append(img_info)

        return cln_infos, nos_infos

    '''load all image in RAM and do justify, based on slide window'''

    '''def pre_processing(self, cln_level: float = 1, nos_level: float = 3):
        patch_size = 128
        stride = 64
        cln_num = 0
        nos_num = 0
        cln_patches = []
        nos_patches = []
        # length_data = len(self.data)
        for data in self.dataset:
            # im_name = os.path.join(sub_root, file)
            image = cv2.imread(data['img'], cv2.IMREAD_GRAYSCALE)
            # label = cv2.imread(data['ann'], cv2.IMREAD_GRAYSCALE)
            gt_semantic_seg = mmcv.imread(data['ann'], flag='unchanged', backend='pillow')
            img = image[120:-120, 120:-120]
            h, w = img.shape
            for i in range(0, h - patch_size + 1, stride):
                for j in range(0, w - patch_size + 1, stride):
                    patch = img[i: i + patch_size, j: j + patch_size]
                    sobely = cv2.Sobel(patch / 255, cv2.CV_64F, 0, 1, ksize=5)
                    ave_score = np.mean(abs(sobely))
                    if ave_score < cln_level:
                        cln_patch = dict(img=patch)
                        gt_patch = gt_semantic_seg[i: i + patch_size, j: j + patch_size]
                        cln_patch['gt_semantic_seg'] = gt_patch
                        cln_patch['seg_fields'] = []
                        cln_patch['seg_fields'].append('gt_semantic_seg')
                        cln_patches.append(cln_patch)
                        cln_num = cln_num + 1
                        cv2.imwrite(os.path.join(self.data_root, 'gan_dataset', 'clean', str(cln_num) + '.png'), patch)
                    elif ave_score > nos_level:
                        nos_patch = dict(img=patch)
                        gt_patch = gt_semantic_seg[i: i + patch_size, j: j + patch_size]
                        nos_patch['gt_semantic_seg'] = gt_patch
                        nos_patch['seg_fields'] = []
                        nos_patch['seg_fields'].append('gt_semantic_seg')
                        nos_patches.append(nos_patch)
                        nos_num = nos_num + 1
                        cv2.imwrite(os.path.join(self.data_root, 'gan_dataset', 'noise', str(nos_num) + '.png'), patch)

        return cln_patches, nos_patches'''

    '''def get_gt_seg_maps(self):
        """Get ground truth segmentation maps for evaluation."""
        gt_seg_maps = []
        for img_info in self.img_infos:
            gt_seg_map = mmcv.imread(
                img_info['ann'], flag='unchanged', backend='pillow')
            if self.reduce_zero_label:
                # avoid using underflow conversion
                gt_seg_map[gt_seg_map == 0] = 255
                gt_seg_map = gt_seg_map - 1
                gt_seg_map[gt_seg_map == 254] = 255

            gt_seg_maps.append(gt_seg_map)

        return gt_seg_maps'''

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['seg_fields'] = []


if __name__ == "__main__":
    # covid = Covid(root="F:/datasets/Covid19/COVID-19-20_v2/Train/", train=False)
    # loader = covid.__get_valid_loader__(num_workers=0)
    # for batch_idx, tensors in enumerate(loader):
    #     print(1)
    '''
        def __init__(self,
                 pipeline,
                 img_dir,
                 img_suffix='.png',
                 cln_dir='train_cln',
                 cor_dir='train_cor',
                 split=None,
                 data_root=None,
                 test_mode=False,
                 ignore_index=255,
                 reduce_zero_label=False,
                 **kwargs):
    '''
    covid = ScoliosisDataset_gan_m(
        data_root='/mnt/sd2/Semantic_Seg/mmsegmentation_for_3classes/data/scoliosis3classes',
        img_dir='gan_dataset',
        cor_dir='train_cor',
        cln_dir='train_cln',
        # split='train.txt',
        pipeline=[
            dict(type='LoadImageFromFile', color_type='grayscale'),
            dict(type='RandomFlip', flip_ratio=0.5),
            # dict(type='PhotoMetricDistortion'),
            # dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size=(128, 128), pad_val=0, seg_pad_val=255),
            dict(type='DefaultFormatBundle'),
            # dict(type='Collect', keys=['img', 'gt_semantic_seg']
            #      ),
        ],
    )
    loader = DataLoader(covid, num_workers=0, batch_size=1, shuffle=True)
    con_pos = 0
    con_neg = 0
    for batch_idx, (img_tensor, ref_tensor) in enumerate(loader):
        if ref_tensor.size() != img_tensor.size():
            print("Error!, shape inconsistent")
            break
        ref_tensor = ref_tensor.detach().numpy()
        if ref_tensor.max() == 0:
            con_neg += 1
        elif ref_tensor.max() == 1:
            con_pos += 1
    print("pos:" + str(con_pos))
    print("neg:" + str(con_neg))
