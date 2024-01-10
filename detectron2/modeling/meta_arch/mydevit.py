# Copyright (c) Facebook, Inc. and its affiliates.
import math
import random
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from numpy.lib import pad
import torch
from torch import nn
from torch.nn import functional as F
from random import randint
from torch.cuda.amp import autocast


from detectron2.config import configurable, get_cfg
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.structures import ImageList, Instances, Boxes
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n

from ..backbone import Backbone, build_backbone
from ..postprocessing import detector_postprocess
from ..proposal_generator import build_proposal_generator
import warnings
from ..roi_heads import build_roi_heads
from ..matcher import Matcher
from .build import META_ARCH_REGISTRY


from PIL import Image
import copy
from ..backbone.fpn import build_resnet_fpn_backbone

from detectron2.layers.roi_align import ROIAlign
from torchvision.ops.boxes import box_area, box_iou

from torchvision.ops import sigmoid_focal_loss

from detectron2.modeling.roi_heads.fast_rcnn import fast_rcnn_inference
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.structures import Boxes, Instances
from fvcore.nn import giou_loss, smooth_l1_loss
from detectron2.structures.masks import PolygonMasks

from lib.dinov2.layers.block import Block
from lib.regionprop import augment_rois, region_coord_2_abs_coord, abs_coord_2_region_coord, SpatialIntegral

COCO_UNSEEN_CLS = ['fig', 'hazelnut']

# 48 class names in order, obtained from load_coco_json() function
COCO_SEEN_CLS = ['date']

def distance_embed(x, temperature = 10000, num_pos_feats = 128, scale=10.0):
    # x: [bs, n_dist]
    x = x[..., None]
    scale = 2 * math.pi * scale
    dim_t = torch.arange(num_pos_feats)
    dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / num_pos_feats)
    sin_x = x * scale / dim_t.to(x.device)
    emb = torch.stack((sin_x[:, :, 0::2].sin(), sin_x[:, :, 1::2].cos()), dim=3).flatten(2)
    return emb # [bs, n_dist, n_emb]

def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.mean(1).sum() / num_masks

def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks

def interpolate(seq, T, mode='linear', force=False):
    # seq: B x C x L
    if (seq.shape[-1] < T) or force:
        return F.interpolate(seq, T, mode=mode) 
    else:
    #     # assume is sorted ascending order
        return seq[:, :, -T:]
    
def generalized_box_iou(boxes1, boxes2) -> torch.Tensor:
    """
    Generalized IoU from https://giou.stanford.edu/

    The input boxes should be in (x0, y0, x1, y1) format

    Args:
        boxes1: (torch.Tensor[N, 4]): first set of boxes
        boxes2: (torch.Tensor[M, 4]): second set of boxes

    Returns:
        torch.Tensor: a NxM pairwise matrix containing the pairwise Generalized IoU
        for every element in boxes1 and boxes2.
    """
    # degenerate boxes gives inf / nan results
    # so do an early check

    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / (area + 1e-6)

def box_cxcywh_to_xyxy(bbox) -> torch.Tensor:
    """Convert bbox coordinates from (cx, cy, w, h) to (x1, y1, x2, y2)

    Args:
        bbox (torch.Tensor): Shape (n, 4) for bboxes.

    Returns:
        torch.Tensor: Converted bboxes.
    """
    cx, cy, w, h = bbox.unbind(-1)
    new_bbox = [(cx - 0.5 * w), (cy - 0.5 * h), (cx + 0.5 * w), (cy + 0.5 * h)]
    return torch.stack(new_bbox, dim=-1)

def _log_classification_stats(pred_logits, gt_classes):
    num_instances = gt_classes.numel()
    if num_instances == 0:
        return
    pred_classes = pred_logits.argmax(dim=1)
    bg_class_ind = pred_logits.shape[1] - 1

    fg_inds = (gt_classes >= 0) & (gt_classes < bg_class_ind)
    num_fg = fg_inds.nonzero().numel()
    fg_gt_classes = gt_classes[fg_inds]
    fg_pred_classes = pred_classes[fg_inds]

    num_false_negative = (fg_pred_classes == bg_class_ind).nonzero().numel()
    num_accurate = (pred_classes == gt_classes).nonzero().numel()
    fg_num_accurate = (fg_pred_classes == fg_gt_classes).nonzero().numel()

    try:
        storage = get_event_storage()
        storage.put_scalar(f"cls_acc", num_accurate / num_instances)
        if num_fg > 0:
            storage.put_scalar(f"fg_cls_acc", fg_num_accurate / num_fg)
            storage.put_scalar(f"false_neg_ratio", num_false_negative / num_fg)
    except:
        pass

def focal_loss(inputs, targets, gamma=0.5, reduction="mean", bg_weight=0.2, num_classes=None):
    """Inspired by RetinaNet implementation"""
    if targets.numel() == 0 and reduction == "mean":
        return input.sum() * 0.0  # connect the gradient
    
    # focal scaling
    ce_loss = F.cross_entropy(inputs, targets, reduction="none")
    p = F.softmax(inputs, dim=-1)
    p_t = p[torch.arange(p.size(0)).to(p.device), targets]  # get prob of target class
    p_t = torch.clamp(p_t, 1e-7, 1-1e-7) # prevent NaN
    loss = ce_loss * ((1 - p_t) ** gamma)

    # bg loss weight
    if bg_weight >= 0:
        assert num_classes is not None
        loss_weight = torch.ones(loss.size(0)).to(p.device)
        loss_weight[targets == num_classes] = bg_weight
        loss = loss * loss_weight

    if reduction == "mean":
        loss = loss.mean()

    return loss


@META_ARCH_REGISTRY.register()
class DevitNet(nn.Module):
    @property
    def device(self):
        return self.pixel_mean.device

    def offline_preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images. Use detectron2 default processing (pixel mean & std).
        Note: Due to FPN size_divisibility, images are padded by right/bottom border. So FPN is consistent with C4 and GT boxes.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        if (self.input_format == 'RGB' and self.offline_input_format == 'BGR') or \
            (self.input_format == 'BGR' and self.offline_input_format == 'RGB'):
            images = [x[[2,1,0],:,:] for x in images]
        if self.offline_div_pixel:
            images = [((x / 255.0) - self.offline_pixel_mean) / self.offline_pixel_std for x in images]
        else:
            images = [(x - self.offline_pixel_mean) / self.offline_pixel_std for x in images]
        images = ImageList.from_tensors(images, self.offline_backbone.size_divisibility)
        return images

    def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images. Use CLIP default processing (pixel mean & std).
        Note: Due to FPN size_divisibility, images are padded by right/bottom border. So FPN is consistent with C4 and GT boxes.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        if self.div_pixel:
            images = [((x / 255.0) - self.pixel_mean) / self.pixel_std for x in images]
        else:
            images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    @staticmethod
    def _postprocess(instances, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image in zip(
            instances, batched_inputs):
            height = input_per_image["height"]  # original image size, before resizing
            width = input_per_image["width"]  # original image size, before resizing
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results
        
    @configurable
    def __init__(self,
                backbone: Backbone,
                offline_backbone: Backbone,
                offline_proposal_generator: nn.Module, 
                pixel_mean: Tuple[float],
                pixel_std: Tuple[float],
                offline_pixel_mean: Tuple[float],
                offline_pixel_std: Tuple[float],
                offline_input_format: Optional[str] = None,

                class_prototypes_file="",   # 感觉可以把prototypes路径给传进去
                bg_prototypes_file="",
                proposal_matcher = None,
                box2box_transform=None,
                seen_cids = [],
                all_cids = [],
                mask_cids = [],

                bg_cls_weight=0.2,
                roialign_size=7,
                box_noise_scale=1.0,
                T_length=128,
                num_cls_layers=3,
                smooth_l1_beta=0.0,
                test_score_thresh=0.001,
                test_nms_thresh=0.5,
                test_topk_per_image=100,
                cls_temp=0.1,              
                num_sample_class=-1,
                batch_size_per_image=128,
                pos_ratio=0.25,
                mult_rpn_score=False,
                only_train_mask=True,
                use_mask=True,
                vit_feat_name=None
                ):
        super().__init__()

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        self.backbone = backbone # Modify ResNet
        self.bg_cls_weight = bg_cls_weight

        if np.sum(pixel_mean) < 3.0: # converrt pixel value to range [0.0, 1.0] by dividing 255.0
            # assert input_format == 'RGB'
            self.div_pixel = True
        else:
            self.div_pixel = False

        # RPN related 
        self.input_format = "RGB"
        self.offline_backbone = offline_backbone
        self.offline_proposal_generator = offline_proposal_generator        
        if offline_input_format and offline_pixel_mean and offline_pixel_std:
            self.offline_input_format = offline_input_format
            self.register_buffer("offline_pixel_mean", torch.tensor(offline_pixel_mean).view(-1, 1, 1), False)
            self.register_buffer("offline_pixel_std", torch.tensor(offline_pixel_std).view(-1, 1, 1), False)
            if np.sum(offline_pixel_mean) < 3.0: # converrt pixel value to range [0.0, 1.0] by dividing 255.0
                assert offline_input_format == 'RGB'
                self.offline_div_pixel = True
            else:
                self.offline_div_pixel = False
        
        self.proposal_matcher = proposal_matcher
        
        # class_prototypes_file
        #  prototypes, class_order_for_inference
        dct = torch.load(class_prototypes_file)
        prototypes = dct['prototypes']
        prototype_label_names = dct['label_names']

        if len(prototypes.shape) == 3:
            class_weights = F.normalize(prototypes.mean(dim=1), dim=-1)
        else:
            class_weights = F.normalize(prototypes, dim=-1)

        for c in all_cids:
            if c not in prototype_label_names:
                prototype_label_names.append(c)
                mask_cids.append(c)
                class_weights = torch.cat([class_weights, torch.zeros(1, class_weights.shape[-1])], dim=0)
        
        train_class_order = [prototype_label_names.index(c) for c in seen_cids]
        test_class_order = [prototype_label_names.index(c) for c in all_cids]

        self.label_names = prototype_label_names

        assert -1 not in train_class_order and -1 not in test_class_order

        self.register_buffer("train_class_weight", class_weights[torch.as_tensor(train_class_order).type(torch.long)])
        self.register_buffer("test_class_weight", class_weights[torch.as_tensor(test_class_order).type(torch.long)])
        self.test_class_order = test_class_order
        
        self.num_train_classes = len(seen_cids)
        self.num_classes = len(all_cids)

        self.all_labels = all_cids
        self.seen_labels = seen_cids

        self.train_class_mask = None
        self.test_class_mask = None

        if len(mask_cids) > 0:
            self.train_class_mask = torch.as_tensor([c in mask_cids for c in seen_cids])
            if self.train_class_mask.sum().item() == 0:
                self.train_class_mask = None

            self.test_class_mask = torch.as_tensor([c in mask_cids for c in all_cids])

        bg_protos = torch.load(bg_prototypes_file)
        if isinstance(bg_protos, dict):  # NOTE: connect to dict output of `generate_prototypes`
            bg_protos = bg_protos['prototypes']
        if len(bg_protos.shape) == 3:
            bg_protos = bg_protos.flatten(0, 1)
        self.register_buffer("bg_tokens", bg_protos)
        self.num_bg_tokens = len(self.bg_tokens)

        self.roialign_size = roialign_size
        self.roi_align = ROIAlign(roialign_size, 1 / backbone.patch_size, sampling_ratio=-1)
        # input: NCHW, Bx5, output BCKK
        self.box_noise_scale = box_noise_scale


        self.T = T_length
        self.Tpos_emb = 128
        self.Temb = 128
        self.Tbg_emb = 128
        hidden_dim = 256
        # N x C x 14 x 14 -> N x 1 
        self.ce = nn.CrossEntropyLoss()
        self.fc_intra_class = nn.Linear(self.Tpos_emb, self.Temb)
        self.fc_other_class = nn.Linear(self.T, self.Temb)
        self.fc_back_class = nn.Linear(self.num_bg_tokens, self.Tbg_emb)

        cls_input_dim = self.Temb * 2 + self.Tbg_emb
        bg_input_dim = self.Temb + self.Tbg_emb

        self.fc_bg_class = nn.Linear(self.T, self.Temb)

        self.box2box_transform = box2box_transform
        self.smooth_l1_beta = smooth_l1_beta
        self.test_score_thresh = test_score_thresh
        self.test_nms_thresh = test_nms_thresh
        self.test_topk_per_image = test_topk_per_image

        self.reg_roialign_size = 20
        self.reg_roi_align = ROIAlign(self.reg_roialign_size, 1 / backbone.patch_size, sampling_ratio=-1)

        reg_feat_dim = self.Temb * 2

        self.rp1 = nn.Sequential(
            nn.Conv2d(reg_feat_dim + 1, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
        )
        self.rp1_out = nn.Conv2d(hidden_dim, 1, kernel_size=3, stride=1, padding=1)

        self.rp2 = nn.Sequential(
            nn.Conv2d(reg_feat_dim + 2, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
        )
        self.rp2_out = nn.Conv2d(hidden_dim, 1, kernel_size=3, stride=1, padding=1)

        self.rp3 = nn.Sequential(
            nn.Conv2d(reg_feat_dim + 3, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
        )
        self.rp3_out = nn.Conv2d(hidden_dim, 1, kernel_size=3, stride=1, padding=1)

        self.rp4 = nn.Sequential(
            nn.Conv2d(reg_feat_dim + 4, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
        )
        self.rp4_out = nn.Conv2d(hidden_dim, 1, kernel_size=3, stride=1, padding=1)
        
        self.rp5 = nn.Sequential(
            nn.Conv2d(reg_feat_dim + 5, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
        )
        self.rp5_out = nn.Conv2d(hidden_dim, 1, kernel_size=3, stride=1, padding=1)

        self.r2c = SpatialIntegral(self.reg_roialign_size)

        self.reg_intra_dist_emb = nn.Linear(self.Tpos_emb, self.Temb)
        self.reg_bg_dist_emb = nn.Linear(self.num_bg_tokens, self.Temb)

        self.cls_temp = cls_temp
        self.evaluation_shortcut = False
        self.num_sample_class = num_sample_class
        self.batch_size_per_image = batch_size_per_image
        self.pos_ratio = pos_ratio
        self.mult_rpn_score = mult_rpn_score

        # ---------- mask related layers --------- # 
        self.only_train_mask = only_train_mask if use_mask else False
        self.use_mask = use_mask

        self.vit_feat_name = vit_feat_name

        if self.use_mask:

            self.mask_roialign_size = 14
            self.mask_roi_align = ROIAlign(self.mask_roialign_size, 1 / backbone.patch_size, sampling_ratio=-1)

            self.mask_intra_dist_emb = nn.Linear(self.Tpos_emb, self.Temb)
            self.mask_bg_dist_emb = nn.Linear(self.num_bg_tokens, self.Temb)

            num_mask_regression_layers = 5
            self.use_init_mask = True

            layer_start_offset = 1 if self.use_init_mask else 0
            self.use_mask_feat_input = True
            self.use_mask_dropout = True
            self.use_mask_inst_norm = True
            self.use_focal_mask = True
            self.use_mask_ms_feat = True

            feat_inp_dim = 256 if self.use_mask_feat_input else 0

            if self.use_mask_feat_input:
                if self.use_mask_ms_feat:
                    self.mask_feat_compress = nn.ModuleList([nn.Conv2d(self.train_class_weight.shape[-1], feat_inp_dim, 1, 1, 0)
                                                            for _ in range(3)])
                    feat_inp_dim = feat_inp_dim * 3
                else:
                    self.mask_feat_compress = nn.Conv2d(self.train_class_weight.shape[-1], feat_inp_dim, 1, 1, 0)
            
            if self.use_init_mask:
                self.fc_init_mask = nn.Conv2d(1, 1, 1, 1, 0)   
                self.fc_init_mask.weight.data.fill_(1.0)
                self.fc_init_mask.bias.data.fill_(0.0)
            
            if self.use_mask_dropout:
                self.mask_dropout = nn.Dropout2d(p=0.5)
            
            hidden_dim = 384

            self.mp_layers = nn.ModuleList([
                nn.Sequential(
                nn.Conv2d(((self.Temb * 2 + feat_inp_dim) if i == 0 else hidden_dim) + i + layer_start_offset, hidden_dim, 
                        kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm2d(hidden_dim, affine=True) if self.use_mask_inst_norm else nn.BatchNorm2d(hidden_dim),
                nn.ReLU(),
            ) for i in range(num_mask_regression_layers)])
            self.mp_out_layers = nn.ModuleList([
                nn.Conv2d(hidden_dim, 1, kernel_size=3, stride=1, padding=1)
                for i in range(num_mask_regression_layers)
            ])
            self.mask_deconv = nn.Sequential(
                nn.ConvTranspose2d(hidden_dim + num_mask_regression_layers + layer_start_offset, 
                                hidden_dim, kernel_size=2, stride=2, padding=0),
                nn.InstanceNorm2d(hidden_dim, affine=True) if self.use_mask_inst_norm else nn.BatchNorm2d(hidden_dim),
                nn.ReLU()
            )
            self.mask_predictor = nn.Conv2d(hidden_dim, 1, kernel_size=1, stride=1, padding=0)

            if self.only_train_mask:
                self.turn_off_box_training(force=True)
                self.turn_off_cls_training(force=True)

    def turn_off_cls_training(self, force=False):
        self._turn_off_modules([
            self.fc_intra_class,
            self.fc_other_class,
            self.fc_back_class,
            self.fc_bg_class
        ], force)

    def turn_off_box_training(self, force=False):
        self._turn_off_modules([
            self.reg_intra_dist_emb,
            self.reg_bg_dist_emb,
            self.r2c,
            self.rp1,
            self.rp1_out,
            self.rp2,
            self.rp2_out,
            self.rp3,
            self.rp3_out,
            self.rp4,
            self.rp4_out,
            self.rp5,
            self.rp5_out,
        ], force)

    def _turn_off_modules(self, modules, force):
        for m in modules:
            if m.training or force: 
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False
              
    @classmethod
    def from_config(cls, cfg, use_bn=False):
        offline_cfg = get_cfg()
        offline_cfg.merge_from_file(cfg.DE.OFFLINE_RPN_CONFIG)
        if cfg.DE.OFFLINE_RPN_LSJ_PRETRAINED: # large-scale jittering (LSJ) pretrained RPN
            offline_cfg.MODEL.BACKBONE.FREEZE_AT = 0 # make all fronzon layers to "SyncBN"
            offline_cfg.MODEL.RESNETS.NORM = "BN" # 5 resnet layers
            offline_cfg.MODEL.FPN.NORM = "BN" # fpn layers
            # offline_cfg.MODEL.RESNETS.NORM = "SyncBN" # 5 resnet layers
            # offline_cfg.MODEL.FPN.NORM = "SyncBN" # fpn layers
            offline_cfg.MODEL.RPN.CONV_DIMS = [-1, -1] # rpn layers
        if cfg.DE.OFFLINE_RPN_NMS_THRESH:
            offline_cfg.MODEL.RPN.NMS_THRESH = cfg.DE.OFFLINE_RPN_NMS_THRESH  # 0.9
        if cfg.DE.OFFLINE_RPN_POST_NMS_TOPK_TEST:
            offline_cfg.MODEL.RPN.POST_NMS_TOPK_TEST = cfg.DE.OFFLINE_RPN_POST_NMS_TOPK_TEST # 1000

        # create offline backbone and RPN
        offline_backbone = build_backbone(offline_cfg)
        offline_rpn = build_proposal_generator(offline_cfg, offline_backbone.output_shape())

        # convert to evaluation mode
        for p in offline_backbone.parameters(): p.requires_grad = False
        for p in offline_rpn.parameters(): p.requires_grad = False
        offline_backbone.eval()
        offline_rpn.eval()

        backbone = build_backbone(cfg)
        for p in backbone.parameters(): p.requires_grad = False
        backbone.eval()

        if cfg.DE.OUT_INDICES:
            vit_feat_name = f'res{cfg.DE.OUT_INDICES[-1]}'
        else:
            vit_feat_name = f'res{backbone.n_blocks - 1}'

        return {
            "backbone": backbone,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "class_prototypes_file": cfg.DE.CLASS_PROTOTYPES,
            "bg_prototypes_file": cfg.DE.BG_PROTOTYPES,

            "roialign_size": cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION,

            "offline_backbone": offline_backbone,
            "offline_proposal_generator": offline_rpn, 
            "offline_input_format": offline_cfg.INPUT.FORMAT if offline_cfg else None,
            "offline_pixel_mean": offline_cfg.MODEL.PIXEL_MEAN if offline_cfg else None,
            "offline_pixel_std": offline_cfg.MODEL.PIXEL_STD if offline_cfg else None,
            
            "proposal_matcher": Matcher(
                cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS,
                cfg.MODEL.ROI_HEADS.IOU_LABELS,
                allow_low_quality_matches=False,
            ),

            # regression
            "box2box_transform": Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS),
            "smooth_l1_beta"        : cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA,
            "test_score_thresh"     : cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST,
            "test_nms_thresh"       : cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST,
            "test_topk_per_image"   : cfg.TEST.DETECTIONS_PER_IMAGE,

            "box_noise_scale": 0.5,

            "cls_temp": cfg.DE.TEMP,
            
            "num_sample_class": cfg.DE.TOPK,
            
            "T_length": cfg.DE.T,
            
            "batch_size_per_image": cfg.DE.RCNN_BATCH_SIZE,
            "pos_ratio": cfg.DE.POS_RATIO,
            
            "mult_rpn_score": cfg.DE.MULTIPLY_RPN_SCORE,

            "num_cls_layers": cfg.DE.NUM_CLS_LAYERS,
            
            "only_train_mask": cfg.DE.ONLY_TRAIN_MASK,
            "use_mask": cfg.MODEL.MASK_ON,
            
            "vit_feat_name": vit_feat_name
        }
    
    def mask_forward(self, features, boxes, class_labels, class_weights, gt_masks=None, feature_dict=None):
          pass
  


    
    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        print('hhhhhhh', len(batched_inputs))
        bs = len(batched_inputs)
        loss_dict = {}
        if not self.training: assert bs == 1

        if self.training:    # 删除self.use_one_shot
            class_weights = self.train_class_weight
        else:
            class_weights = self.test_class_weight

        num_classes = len(class_weights)

        # Online Learning需要具备下述的几个特点：数据从流式数据源获取，比如Kafka、MQ
        # 所以我们训练都叫离线offline的
        with torch.no_grad():
            # with autocast(enabled=True):
            if self.offline_backbone.training or self.offline_proposal_generator.training:  
                self.offline_backbone.eval() 
                self.offline_proposal_generator.eval()  
            images = self.offline_preprocess_image(batched_inputs)
            features = self.offline_backbone(images.tensor)
            proposals, _ = self.offline_proposal_generator(images, features, None)     
            images = self.preprocess_image(batched_inputs)

        with torch.no_grad():
            if self.backbone.training: self.backbone.eval()
            with autocast(enabled=True):
                all_patch_tokens = self.backbone(images.tensor)
                patch_tokens = all_patch_tokens[self.vit_feat_name]
                all_patch_tokens.pop(self.vit_feat_name)
                # patch_tokens = self.backbone(images.tensor)['res11'] 

        if self.training: 
            with torch.no_grad():
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                gt_boxes = [x.gt_boxes.tensor for x in gt_instances]

                rpn_boxes = [x.proposal_boxes.tensor for x in proposals]
                # could try to use only gt_boxes to see the accuracy
                if self.training:
                    noisy_boxes = self.prepare_noisy_boxes(gt_boxes, images.tensor.shape)
                    boxes = [torch.cat([gt_boxes[i], noisy_boxes[i], rpn_boxes[i]]) 
                            for i in range(len(batched_inputs))]
                else:
                    boxes = rpn_boxes

                class_labels = []
                matched_gt_boxes = []
                resampled_proposals = []

                num_bg_samples, num_fg_samples = [], []
                gt_masks = []

                for proposals_per_image, targets_per_image in zip(boxes, gt_instances):
                    match_quality_matrix = box_iou(
                        targets_per_image.gt_boxes.tensor, proposals_per_image
                    ) # (N, M)
                    matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
                    if len(targets_per_image.gt_classes) > 0:
                        class_labels_i = targets_per_image.gt_classes[matched_idxs]
                    else:
                        # no annotation on this image
                        assert torch.all(matched_labels == 0)
                        class_labels_i = torch.zeros_like(matched_idxs)
                    class_labels_i[matched_labels == 0] = num_classes
                    class_labels_i[matched_labels == -1] = -1
                    
        #             if self.training or self.evaluation_shortcut:
        #                 positive = ((class_labels_i != -1) & (class_labels_i != num_classes)).nonzero().flatten()
        #                 negative = (class_labels_i == num_classes).nonzero().flatten()

        #                 batch_size_per_image = self.batch_size_per_image # 512
        #                 num_pos = int(batch_size_per_image * self.pos_ratio)
        #                 # protect against not enough positive examples
        #                 num_pos = min(positive.numel(), num_pos)
        #                 num_neg = batch_size_per_image - num_pos
        #                 # protect against not enough negative examples
        #                 num_neg = min(negative.numel(), num_neg)

        #                 perm1 = torch.randperm(positive.numel(), device=self.device)[:num_pos]
        #                 perm2 = torch.randperm(negative.numel())[:num_neg].to(self.device) # torch.randperm(negative.numel(), device=negative.device)[:num_neg]
        #                 pos_idx = positive[perm1]
        #                 neg_idx = negative[perm2]
        #                 sampled_idxs = torch.cat([pos_idx, neg_idx], dim=0)
        #             else:
        #                 sampled_idxs = torch.arange(len(proposals_per_image), device=self.device).long()

        #             proposals_per_image = proposals_per_image[sampled_idxs]
        #             class_labels_i = class_labels_i[sampled_idxs]
                    
        #             if len(targets_per_image.gt_boxes.tensor) > 0:
        #                 gt_boxes_i = targets_per_image.gt_boxes.tensor[matched_idxs[sampled_idxs]]
        #                 if self.use_mask:
        #                     gt_masks_i = targets_per_image.gt_masks[matched_idxs[sampled_idxs]]
        #             else:
        #                 gt_boxes_i = torch.zeros(len(sampled_idxs), 4, device=self.device) # not used anyway
        #                 if self.use_mask:
        #                     gt_masks_i = PolygonMasks([[np.zeros(6)],] * len(sampled_idxs)).to(self.device)

        #             resampled_proposals.append(proposals_per_image)
        #             class_labels.append(class_labels_i)
        #             matched_gt_boxes.append(gt_boxes_i)
        #             if self.use_mask:
        #                 gt_masks.append(gt_masks_i)

        #             num_bg_samples.append((class_labels_i == num_classes).sum().item())
        #             num_fg_samples.append(class_labels_i.numel() - num_bg_samples[-1])
                
        #         if self.training:
        #             storage = get_event_storage()
        #             storage.put_scalar("fg_count", np.mean(num_fg_samples))
        #             storage.put_scalar("bg_count", np.mean(num_bg_samples))

        #         class_labels = torch.cat(class_labels)
        #         matched_gt_boxes = torch.cat(matched_gt_boxes) # for regression purpose.
        #         if self.use_mask:
        #             gt_masks = PolygonMasks.cat(gt_masks)
                
        #         rois = []
        #         for bid, box in enumerate(resampled_proposals):
        #             batch_index = torch.full((len(box), 1), fill_value=float(bid)).to(self.device) 
        #             rois.append(torch.cat([batch_index, box], dim=1))
        #         rois = torch.cat(rois)
        # else:
        #     boxes = proposals[0].proposal_boxes.tensor 
        #     rois = torch.cat([torch.full((len(boxes), 1), fill_value=0).to(self.device) , 
        #                     boxes], dim=1)

        # roi_features = self.roi_align(patch_tokens, rois) # N, C, k, k
        # roi_bs = len(roi_features)
