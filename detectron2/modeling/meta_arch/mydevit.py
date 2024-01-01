import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers.roi_align import ROIAlign
from detectron2.config import configurable
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.layers import move_device_like
from detectron2.structures import ImageList, Instances
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n

from ..backbone import Backbone, build_backbone
from ..postprocessing import detector_postprocess
from ..proposal_generator import build_proposal_generator
from ..roi_heads import build_roi_heads
from .build import META_ARCH_REGISTRY




@META_ARCH_REGISTRY.register()
class DevitNet(nn.Module):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation = RPN, Region Proposal Network
    3. Per-region feature extraction and prediction
    """
    @configurable
    def __init__(
        self,
        backbone: Backbone,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        class_prototypes_file="",
        roialign_size=7,
        box_noise_scale=1.0,
        input_format: Optional[str] = None,
        vis_period: int = 0,
        vit_feat_name = None
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super().__init__()
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads

        self.input_format = input_format
        self.vis_period = vis_period
        self.vit_feat_name = vit_feat_name
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        
        dct = torch.load(class_prototypes_file)
        prototypes = dct['prototypes']
        prototype_label_names = dct['label_names']

        if len(prototypes.shape) == 3:
            class_weights = F.normalize(prototypes.mean(dim=1), dim=-1)
        else:
            class_weights = F.normalize(prototypes, dim=-1)

        train_class_order = [prototype_label_names.index(c) for c in ['fig', 'hazelnut']]
        test_class_order = [prototype_label_names.index(c) for c in ['date', 'fig', 'hazelnut']]
        self.register_buffer("train_class_weight", class_weights[torch.as_tensor(train_class_order)])
        self.register_buffer("test_class_weight", class_weights[torch.as_tensor(test_class_order)])

        self.roialign_size = roialign_size
        self.roi_align = ROIAlign(roialign_size, 1 / backbone.patch_size, sampling_ratio=-1)
        # input: NCHW, Bx5, output BCKK
        self.box_noise_scale = box_noise_scale

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        vit_feat_name = f'res{backbone.n_blocks - 1}'
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "roialign_size": cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION,
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "vit_feat_name": vit_feat_name
        }
    
    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        # ============step1============
        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
        # eg: {'p2': torch.Size([b, c, 200, 304]), 'p3': torch.Size([b, c, 100, 152]), 'p4': torch.Size([b, c, 50, 76]), 'p5': torch.Size([b, c, 25, 38]), 'p6': torch.Size([b, c, 13, 19])}

        # ============step2=============提取特征的backbone
        features = self.backbone(images.tensor)
        print("&&&&&&&&&&&&", features, features.size())
        patch_tokens = features[self.vit_feat_name]
        print("************", patch_tokens, patch_tokens.size())

        # ============step3=============RPN
        if self.proposal_generator is not None:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}
        
        # =============ROIAlign===========
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

                resampled_proposals = []
                for proposals_per_image, targets_per_image in zip(boxes, gt_instances):
                    resampled_proposals.append(proposals_per_image)

                rois = []
                for bid, box in enumerate(resampled_proposals):
                    batch_index = torch.full((len(box), 1), fill_value=float(bid)).to(self.device) 
                    rois.append(torch.cat([batch_index, box], dim=1))
                rois = torch.cat(rois)
        else:
            boxes = proposals[0].proposal_boxes.tensor 
            rois = torch.cat([torch.full((len(boxes), 1), fill_value=0).to(self.device) , 
                            boxes], dim=1)        

        roi_features = self.roi_align(patch_tokens, rois) # N, C, k, k
        roi_bs = len(roi_features)

        if self.training:
            class_weights = self.train_class_weight  
        else:
            class_weights = self.test_class_weight 

        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses
    
    def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [self._move_to_current_device(x["image"]) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(
            images,
            self.backbone.size_divisibility,
            padding_constraints=self.backbone.padding_constraints,
        )
        return images
    
    def visualize_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 top-scoring predicted
        object proposals on the original image. Users can implement different
        visualization functions for different models.

        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch

    @property
    def device(self):
        return self.pixel_mean.device

    def _move_to_current_device(self, x):
        return move_device_like(x, self.pixel_mean)
