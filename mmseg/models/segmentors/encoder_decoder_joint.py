import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.core import add_prefix
from mmseg.ops import resize
from .. import builder
from ..builder import SEGMENTORS
from .base import BaseSegmentor


@SEGMENTORS.register_module()
class EncoderDecoder_joint(BaseSegmentor):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 backbone,
                 backbone_gan,
                 decode_head,
                 feature_selection=None,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 generator_head=None
                 ):
        super(EncoderDecoder_joint, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        self.backbone_gan = builder.build_backbone(backbone_gan)
        self.with_feature_selection = False
        if neck is not None:
            self.neck = builder.build_neck(neck)
        if feature_selection is not None:
            self.feature_selection = builder.build_feature_selection(feature_selection)
            self.with_feature_selection = True
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)
        self.G_head = builder.build_generator_head(generator_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        '''build discriminator'''
        self.disc_steps = 1 if self.train_cfg is None else self.train_cfg.get(
            'disc_steps', 1)
        self.disc_init_steps = (0 if self.train_cfg is None else
                                self.train_cfg.get('disc_init_steps', 0))
        if self.train_cfg is None:
            self.direction = ('a2b' if self.test_cfg is None else
                              self.test_cfg.get('direction', 'a2b'))
        else:
            self.direction = self.train_cfg.get('direction', 'a2b')
        self.step_counter = 0  # counting training steps

        # self.discriminator = builder.build_component(discriminator)
        self.init_weights(pretrained=pretrained)

        assert self.with_decode_head

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        self.decode_head = builder.build_head(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

    def _init_auxiliary_head(self, auxiliary_head):
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(builder.build_head(head_cfg))
            else:
                self.auxiliary_head = builder.build_head(auxiliary_head)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone and heads.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        super(EncoderDecoder_gan, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        self.backbone_gan.init_weights(pretrained=pretrained)
        self.decode_head.init_weights()
        # init GAN
        # self.discriminator.init_weight()
        # self.G_head.init_weights()
        if self.with_auxiliary_head:
            if isinstance(self.auxiliary_head, nn.ModuleList):
                for aux_head in self.auxiliary_head:
                    aux_head.init_weights()
            else:
                self.auxiliary_head.init_weights()

    def extract_feat(self, img):
        """Extract features from images."""
        x = self.backbone(img)
        y = self.backbone_gan(img)
        if self.with_feature_selection:
            x, y = self.feature_selection(x, y)
        if self.with_neck:
            x = self.neck(x)
        return x, y

    # need correct
    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x, y = self.extract_feat(img)
        out = self._decode_head_forward_test(x, img_metas)
        out_gen = self.G_head(y, img)
        # out_gen = resize(
        #     input=out_gen,
        #     size=img.shape[2:],
        #     mode='bilinear',
        #     align_corners=self.align_corners)

        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out, out_gen

    def _decode_head_forward_train(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.forward_train(x, img_metas,
                                                     gt_semantic_seg,
                                                     self.train_cfg)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _decode_head_forward_test(self, x, img_metas):
        """Run forward function and calculate loss for decode head in
        inference."""
        seg_logits = self.decode_head.forward_test(x, img_metas, self.test_cfg)
        return seg_logits

    def _auxiliary_head_forward_train(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.forward_train(x, img_metas,
                                                  gt_semantic_seg,
                                                  self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.forward_train(
                x, img_metas, gt_semantic_seg, self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses

    def forward_dummy(self, img):
        """Dummy forward function."""
        seg_logit = self.encode_decode(img, None)

        return seg_logit

    def forward_train(self, img, img_metas, gt_semantic_seg):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        x, y = self.extract_feat(img)

        losses = dict()

        loss_decode = self._decode_head_forward_train(x, img_metas,
                                                      gt_semantic_seg)
        losses.update(loss_decode)
        gen_out = self.G_head(y, img)  # the output of generators
        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                x, img_metas, gt_semantic_seg)
            losses.update(loss_aux)

        return losses, gen_out

    # TODO refactor
    def slide_inference(self, img, img_meta, rescale):
        """Inference by sliding-window with overlap."""

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = img.size()
        assert h_crop <= h_img and w_crop <= w_img, (
            'crop size should not greater than image size')
        num_classes = self.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_seg_logit = self.encode_decode(crop_img, img_meta)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            # cast count_mat to constant while exporting to ONNX
            count_mat = torch.from_numpy(
                count_mat.cpu().detach().numpy()).to(device=img.device)
        preds = preds / count_mat
        if rescale:
            preds = resize(
                preds,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
        return preds

    def whole_inference(self, img, img_meta, rescale):
        """Inference with full image."""
        if not isinstance(img_meta, list):
            img_meta = img_meta.data
        seg_logit, gen_out = self.encode_decode(img, img_meta)
        if rescale:
            seg_logit = resize(
                seg_logit,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)

            gen_out = resize(
                gen_out,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)

        return seg_logit, gen_out

    def inference(self, img, img_meta, rescale):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """
        if not isinstance(img_meta, list):
            img_meta = img_meta.data[0]
        assert self.test_cfg.mode in ['slide', 'whole']
        # if isinstance(img_meta, list):
        #     ori_shape = img_meta[0][0]['ori_shape']
        # else:
        #     ori_shape = img_meta[0][0]['ori_shape']
        # assert all(_[0]['ori_shape'] == ori_shape for _ in img_meta.data)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(img, img_meta, rescale)
        else:
            seg_logit, gen_out = self.whole_inference(img, img_meta, rescale)
        output = F.softmax(seg_logit, dim=1)
        flip = img_meta[0]['flip']
        if flip:
            flip_direction = img_meta[0]['flip_direction']
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                output = output.flip(dims=(3,))
                gen_out = gen_out.flip(dims=(3,))
            elif flip_direction == 'vertical':
                output = output.flip(dims=(2,))
                gen_out = gen_out.flip(dims=(2,))

        return output, gen_out

    def simple_test(self, img, img_meta, rescale=True, return_logist=False):
        """Simple test with single image."""
        seg_logit, gen_out = self.inference(img, img_meta, rescale) #
        seg_pred = seg_logit.argmax(dim=1)
        if torch.onnx.is_in_onnx_export():
            return seg_pred, gen_out
        seg_pred = seg_pred.cpu().numpy()
        gen_out = gen_out
        # unravel batch dim
        if return_logist:
            # seg_logit = seg_logit.cpu()
            return seg_logit, gen_out
        seg_pred = list(seg_pred)
        gen_out = list(gen_out)
        return seg_pred, gen_out

    def aug_test(self, imgs, img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # print(imgs.shape())
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit, gen_out = self.inference(imgs[0].cuda(), img_metas[0], rescale)
        for i in range(1, len(imgs)):
            cur_seg_logit, _ = self.inference(imgs[i].cuda(), img_metas[i], rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(imgs)
        seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred, gen_out


    def parse_loss(self, losses):
        loss, log_vars = self._parse_losses(losses)
        return loss, log_vars

    def train_step(self, data_batch, optimizer, **kwargs):
        """Training step function.

        Args:
            data_batch (dict): Dict of the input data batch.
            optimizer (dict[torch.optim.Optimizer]): Dict of optimizers for
                the generators and discriminator.

        Returns:
            dict: Dict of loss, information for logger, the number of samples\
                and results for visualization.
        """
        """
        losses = self(**data_batch)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(data_batch['img'].data))

        return outputs
        """
        losses = self(**data_batch)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(data_batch['img'].data))
        return outputs
        # # data
        # img = data_batch['img']
        # # img_b = data_batch['img_metas']
        # meta = data_batch['img_metas']
        # image_seg_gt = data_batch['gt_semantic_seg']
        # # forward generators
        # outputs = self.forward_train(img, meta, image_seg_gt)
        # # outputs = self.forward(img_a, img_b, meta, return_loss=True)
        # log_vars = dict()
        #
        # # discriminator
        # set_requires_grad(self.discriminator, True)
        # # optimize
        # optimizer['discriminator'].zero_grad()
        # log_vars.update(self.backward_discriminator(outputs=outputs))
        # optimizer['discriminator'].step()
        #
        # # generators, no updates to discriminator parameters.
        # if (self.step_counter % self.disc_steps == 0
        #         and self.step_counter >= self.disc_init_steps):
        #     set_requires_grad(self.discriminator, False)
        #     # optimize
        #     optimizer['generators'].zero_grad()
        #     log_vars.update(self.backward_generator(outputs=outputs))
        #     optimizer['generators'].step()
        #
        # self.step_counter += 1
        #
        # log_vars.pop('loss', None)  # remove the unnecessary 'loss'
        # results = dict(
        #     log_vars=log_vars,
        #     num_samples=len(outputs['real_a']),
        #     results=dict(
        #         real_a=outputs['real_a'].cpu(),
        #         fake_b=outputs['fake_b'].cpu(),
        #         real_b=outputs['real_b'].cpu()))
        #
        # return results

    '''def val_step(self, data_batch, **kwargs):
        """Validation step function.

        Args:
            data_batch (dict): Dict of the input data batch.
            kwargs (dict): Other arguments.

        Returns:
            dict: Dict of evaluation results for validation.
        """
        # data
        img_a = data_batch['img_a']
        img_b = data_batch['img_b']
        meta = data_batch['meta']

        # forward generators
        results = self.forward(img_a, img_b, meta, test_mode=True, **kwargs)
        return results'''
    def forward_test(self, imgs, img_metas, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch.
        """


        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got '
                                f'{type(var)}')
        # img_metas = img_metas[0].data  # temporary
        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(imgs)}) != '
                             f'num of image meta ({len(img_metas)})')
        # all images in the same aug batch all of the same ori_shape and pad
        # shape
        # print(img_metas.shape())
        if isinstance(img_metas[0], list):
            for img_meta in img_metas:
                ori_shapes = [_['ori_shape'] for _ in img_meta]
                assert all(shape == ori_shapes[0] for shape in ori_shapes)
                img_shapes = [_['img_shape'] for _ in img_meta]
                assert all(shape == img_shapes[0] for shape in img_shapes)
                pad_shapes = [_['pad_shape'] for _ in img_meta]
                assert all(shape == pad_shapes[0] for shape in pad_shapes)

        else:
            for img_meta in img_metas:
                ori_shapes = [_[0]['ori_shape'] for _ in img_meta.data]
                assert all(shape == ori_shapes[0] for shape in ori_shapes)
                img_shapes = [_[0]['img_shape'] for _ in img_meta.data]
                assert all(shape == img_shapes[0] for shape in img_shapes)
                pad_shapes = [_[0]['pad_shape'] for _ in img_meta.data]
                assert all(shape == pad_shapes[0] for shape in pad_shapes)

        if num_augs == 1:
            return self.simple_test(imgs[0].cuda(), img_metas[0], **kwargs)
        else:
            return self.aug_test(imgs, img_metas, **kwargs)

