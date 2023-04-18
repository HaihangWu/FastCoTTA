from matplotlib.pyplot import text
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.core import add_prefix
from mmseg.ops import resize
from mmseg.models import builder
from mmseg.models.builder import SEGMENTORS
from mmseg.models.segmentors.base import BaseSegmentor
from mmseg.models.segmentors.encoder_decoder import EncoderDecoder

from .segclip_untils import tokenize
import numpy as np
import tqdm

import os
import matplotlib.pyplot as plt

class_description=['roads are identified by clear lane markings, road signs, traffic signals for controlling traffic flow, and roadside elements like streetlights and landmarks',
                   'sidewalk is identified by their paths, streetlights for illumination, signage for guidance, and landscape elements such as trees and greenery',
                   'building is identified by diverse architectural styles, varying heights, distinct facade materials, unique window patterns, and signage or logos',
                   'wall is identified by distinct materials (e.g., brick, concrete, or stone), textures, and any murals or artwork present',
                   'fence is identified by material (e.g., wood, metal, or vinyl), the fence height, its design or pattern (e.g., picket, chain-link, or lattice)',
                   'pole is identified by  their material (e.g., wood, metal, or concrete),  shape (e.g., round or square), color, and any attached elements such as street signs, traffic signals, or utility lines',
               'traffic light is identified by color-coded signals (red, yellow, and green), the position of the lights in a vertical or horizontal arrangement, the presence of additional signal lights for turning lanes or pedestrian crossings, and their mounting style, either on poles or suspended from wires',
                   'traffic sign is identified by  When navigating, the visual features of traffic signs include their shape (e.g., rectangular, triangular, or octagonal), color (e.g., red, yellow, or blue), distinct symbols or text, and reflective properties for visibility',
                   'vegetation is identified by  the type and size of plants (e.g., trees, shrubs, or grass), their arrangement or pattern, the presence of distinct or colorful foliage, seasonal changes in appearance, and any landscaped areas or green spaces',
                   'terrain is identified by  the elevation changes (e.g., hills, valleys, or plateaus), surface type and texture (e.g., rocky, sandy, or grassy), the presence of natural features (e.g., rivers, lakes, or forests), and any unique land formations (e.g., cliffs, dunes, or canyons)',
                   'sky is identified by the position and movement of celestial bodies (e.g., the Sun, Moon, and stars), cloud formations and coverage, changes in color and brightness throughout the day and during different weather conditions, and the presence of atmospheric phenomena (e.g., rainbows, auroras, or sunsets) ',
               'person is identified by their clothing (colors, patterns, and styles), physical appearance (height, hair color, and other distinguishing characteristics), body language and gestures, and any accessories they may carry (e.g., umbrellas, bags, or walking aids).',
                   'rider is identified by  the type of vehicle being ridden (e.g., bicycle, motorcycle, or horse), the rider clothing (e.g., helmets, high-visibility vests, or protective gear), their position on the road or path, and any accessories or equipment related to their vehicle (e.g., saddlebags, lights, or reflectors)',
                   'car is identified by  the vehicle size, shape, and color, the make and model, distinct design elements (e.g., headlights, taillights, or grilles), the presence of any decals or logos, and the car position on the road.  ',
                   'truck is identified by the vehicle size and shape, its color and any markings (e.g., company logos or identification numbers), the type of cargo or trailer it is carrying, distinct design elements (e.g., headlights, taillights, or grilles), and the truck position on the road. ',
                   'bus is identified by the vehicle size, shape, and color, any distinctive markings or logos (e.g., transit authority or company branding), route information displayed on the front and sides, distinct design elements (e.g., large windows, headlights, or taillights), and the bus position on the road or at designated stops. ',
                   'train is identified by the train size and length, its color and any markings or logos (e.g., railway company or line identification), the type of cars being pulled (e.g., passenger, freight, or specialized), distinct design elements (e.g., headlights, windows, or connecting couplers), and the train position on the tracks or at stations.',
                   'motorcycle is identified by  the vehicle size, shape, and color, the make and model, distinct design elements (e.g., headlights, taillights, or exhaust), the rider clothing and protective gear (e.g., helmets, jackets, or gloves), and the motorcycle position on the road',
               'bicycle is identified by  the bicycle design (e.g., road bike, mountain bike, or cruiser), distinct components (e.g., handlebars, pedals, or tires), the accessories (e.g., baskets, lights, or reflectors), and the cyclist position on the road or bike lane.']


@SEGMENTORS.register_module()
class SegLanguage(EncoderDecoder):
    """Encoder Decoder segmentors.
    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 class_names,
                 text_encoder=None,
                 text_decoder=None,
                 ft_model=False,
                 include_key=None,
                 load_text_embedding=None,
                 #  init_cfg=None,
                 **args):
        super(SegLanguage, self).__init__(**args)
        # if pretrained_text is not None:
        #     assert text_encoder.get('pretrained') is None, \
        #         'both text encoder and segmentor set pretrained weight'
        #     text_encoder.pretrained = pretrained_text

        if  text_encoder:
            self.text_feat=None
            self.text_encoder = builder.build_backbone(text_encoder)
            self.load_text_embedding = load_text_embedding
            self.class_names = class_names

            if not self.load_text_embedding:
                # if not self.multi_prompts:
                    self.texts = torch.cat([tokenize(f"a photo of a {c}") for c in self.class_names])
                    #self.texts = torch.cat([tokenize(f"{c}") for c in class_description])
                    #print(self.texts[0][:10],self.texts[1][:10],self.texts[2][:10])
                # else:
                #     self.texts = self._get_multi_prompts(self.class_names)

        if text_decoder:
            self.text_decode_head = builder.build_head(text_decoder)

        self._freeze_stages(self.text_encoder)
        self._freeze_stages(self.text_decode_head, include_key=include_key)
        if ft_model is False:
            self._freeze_stages(self.backbone)
            self._freeze_stages(self.decode_head)

    def _freeze_stages(self, model, include_key=None):
        """Freeze stages param and norm stats."""
        for n, m in model.named_parameters():
            if include_key is not None and isinstance(include_key, str):
                if include_key in n:
                    m.requires_grad = False
            else:
                m.requires_grad = False



    def text_embedding(self, texts, img):
        text_embeddings = self.text_encoder(texts.to(img.device))
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        return text_embeddings


    def forward_train(self, img, img_metas, gt_semantic_seg):
        visual_feat = self.extract_feat(img)

        if self.text_feat is None:
            if self.load_text_embedding:
                text_feat = np.load(self.load_text_embedding)
                self.text_feat = torch.from_numpy(text_feat).to(img.device)
            else:
                self.text_feat = self.text_embedding(self.texts, img)


        # feat = []
        # feat.append(visual_feat)
        # feat.append(text_feat)

        losses = dict()
        loss_decode = self._decode_head_forward_train(visual_feat, img_metas, gt_semantic_seg)
        losses.update(loss_decode)

        return losses

    def encode_decode(self, img, img_metas):
        #print("image info:", type(img), img.size(), img_metas,img)
        visual_feat = self.extract_feat(img)

        if self.text_feat is None:
            if self.load_text_embedding:
                text_feat = np.load(self.load_text_embedding)
                self.text_feat = torch.from_numpy(text_feat).to(img.device)
            else:
                self.text_feat = self.text_embedding(self.texts, img)
            self.text_decode_head.init_predictor(self.text_feat)


        out = self._decode_head_forward_test(visual_feat, img_metas)
        out = resize(
            input=out,
            #input=out["pred_masks"],
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out

    def _decode_head_forward_train(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        seg_logits_visual = self.decode_head.forward_test(x, img_metas, self.test_cfg)
        seg_logits_text = self.text_decode_head.forward_test(x, img_metas, self.test_cfg)
        gt_semantic_seg = gt_semantic_seg[:,:,:,:,-1]
        if seg_logits_text.dim()!=gt_semantic_seg.dim():
            print("dimension is different:",seg_logits_text.dim(),gt_semantic_seg.dim(),seg_logits_text.size(),gt_semantic_seg.size())
            exit()

        loss_decode = self.decode_head.losses(seg_logits_visual + seg_logits_text, gt_semantic_seg)
        #loss_decode = self.decode_head.losses(seg_logits_text, gt_semantic_seg)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _decode_head_forward_test(self, x, img_metas):
        """Run forward function and calculate loss for decode head in
        inference."""
        seg_logits_visual = self.decode_head.forward_test(x, img_metas, self.test_cfg)
        seg_logits_text = self.text_decode_head.forward_test(x, img_metas, self.test_cfg)
        return seg_logits_text
        #return seg_logits

    # TODO refactor
    # def slide_inference(self, img, img_meta, rescale):
    #     """Inference by sliding-window with overlap.
    #     If h_crop > h_img or w_crop > w_img, the small patch will be used to
    #     decode without padding.
    #     """
    #     h_stride, w_stride = self.test_cfg.stride
    #     h_crop, w_crop = self.test_cfg.crop_size
    #     batch_size, _, h_img, w_img = img.size()
    #     num_classes = len(self.both_class)
    #     h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
    #     w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
    #     preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
    #     count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
    #     for h_idx in range(h_grids):
    #         for w_idx in range(w_grids):
    #             y1 = h_idx * h_stride
    #             x1 = w_idx * w_stride
    #             y2 = min(y1 + h_crop, h_img)
    #             x2 = min(x1 + w_crop, w_img)
    #             y1 = max(y2 - h_crop, 0)
    #             x1 = max(x2 - w_crop, 0)
    #             crop_img = img[:, :, y1:y2, x1:x2]
    #             crop_seg_logit = self.encode_decode(crop_img, img_meta)
    #             preds += F.pad(crop_seg_logit,
    #                            (int(x1), int(preds.shape[3] - x2), int(y1),
    #                             int(preds.shape[2] - y2)))
    #
    #             count_mat[:, :, y1:y2, x1:x2] += 1
    #     assert (count_mat == 0).sum() == 0
    #     if torch.onnx.is_in_onnx_export():
    #         count_mat = torch.from_numpy(
    #             count_mat.cpu().detach().numpy()).to(device=img.device)
    #     preds = preds / count_mat
    #     if rescale:
    #         preds = resize(
    #             preds,
    #             size=img_meta[0]['ori_shape'][:2],
    #             mode='bilinear',
    #             align_corners=self.align_corners,
    #             warning=False)
    #     return preds