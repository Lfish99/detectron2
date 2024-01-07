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
from detectron2.data.datasets.coco_zeroshot_categories import COCO_SEEN_CLS, \
    COCO_UNSEEN_CLS, COCO_OVD_ALL_CLS
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

@META_ARCH_REGISTRY.register()
class DevitNet(nn.Module):
    @configurable
    def __init__(self,
                backbone: Backbone,

                pixel_mean: Tuple[float],
                pixel_std: Tuple[float],

                class_prototypes_file="",
                bg_prototypes_file="",
                roialign_size=7,
                box_noise_scale=1.0,
                proposal_matcher = None,

                box2box_transform=None,
                smooth_l1_beta=0.0,
                test_score_thresh=0.001,
                test_nms_thresh=0.5,
                test_topk_per_image=100,
                cls_temp=0.1,
                
                num_sample_class=-1,
                seen_cids = [],
                all_cids = [],
                mask_cids = [],
                T_length=128,
                
                bg_cls_weight=0.2,
                batch_size_per_image=128,
                pos_ratio=0.25,
                mult_rpn_score=False,
                num_cls_layers=3,
                use_one_shot= False,
                one_shot_reference= '',
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

        self.use_one_shot = use_one_shot

        self.one_shot_ref = None

        if use_one_shot:
            self.one_shot_ref = torch.load(one_shot_reference)
        
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
            self.per_cls_cnn,
            self.bg_cnn,
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
            
            "use_one_shot": cfg.DE.ONE_SHOT_MODE,
            "one_shot_reference": cfg.DE.ONE_SHOT_REFERENCE,
            
            "only_train_mask": cfg.DE.ONLY_TRAIN_MASK,
            "use_mask": cfg.MODEL.MASK_ON,
            
            "vit_feat_name": vit_feat_name
        }
    
    def mask_forward(self, features, boxes, class_labels, class_weights, gt_masks=None, feature_dict=None):
      pass
  


    
    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
      pass
