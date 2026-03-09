import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision import models
from network.CoordAttention import CoordAtt

class resnet(nn.Module):
    def __init__(self,global_fusion):
        super(resnet, self).__init__()
        self.model = models.resnet50() # pretrained=True
        self.model.load_state_dict(torch.load(global_fusion))
        for p in self.model.parameters():
            p.requires_grad = False
        for p in self.model.layer4.parameters():
            p.requires_grad = True

        self.fc1 = nn.Linear(2048, 1024)
        self.bn = nn.BatchNorm1d(1024)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(1024, 512)

    def forward(self, x):
        x = x.permute(0,1,4,2,3).contiguous()
        x_shape = x.shape
        x = x.reshape(x_shape[0]*x_shape[1],x_shape[2],x_shape[3],x_shape[4]).contiguous() # 448*800
        with torch.no_grad():
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)
            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)  # 28*50
        x = self.model.layer4(x)  # 14*25
        x = self.model.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

import torch
import torch.nn.functional as F
import math
from detectron2.utils import comm

import open_clip

from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec

@BACKBONE_REGISTRY.register()
class CLIP(Backbone):
    def __init__(self):
        super().__init__()
        model_name = "convnext_large_d_320"
        pretrained= "laion2b_s29b_b131k_ft_soup"
        # download on local rank 0 first
        if comm.get_local_rank() == 0:
            open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        comm.synchronize()

        self.model_name = model_name
        self.pretrained = pretrained

        self.clip_model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.text_tokenizer = open_clip.get_tokenizer(model_name)

        model_name = model_name.lower()
        if 'convnext_' in model_name:
            self.model_type = 'convnext'
            if '_base' in model_name:
                self.output_channels = [128, 128, 256, 512, 1024]
            elif '_large' in model_name:
                self.output_channels = [192, 192, 384, 768, 1536]
            elif '_xxlarge' in model_name:
                self.output_channels = [384, 384, 768, 1536, 3072]
        
        elif 'rn' in model_name:
            self.model_type = 'resnet'
            if model_name.replace('-quickgelu', '') in ['rn50', 'rn101']:
                self.output_channels = [64, 256, 512, 1024, 2048]
            elif model_name == 'rn50x4':
                self.output_channels = [80, 320, 640, 1280, 2560]
            elif model_name == 'rn50x16':
                self.output_channels = [96, 384, 768, 1536, 3072]
            elif model_name == 'rn50x64':
                self.output_channels = [128, 512, 1024, 2048, 4096]

        self._out_feature_strides = {
            "stem": 2,
            "res2": 4,
            "res3": 8,
            "res4": 16,
            "res5": 32,
            "clip_embedding": -1
        }
        self._out_feature_channels = {
            "stem": self.output_channels[0],
            "res2": self.output_channels[1],
            "res3": self.output_channels[2],
            "res4": self.output_channels[3],
            "res5": self.output_channels[4],
            "clip_embedding": self.dim_latent
        }

        self.eval()
        self.freeze_everything()

    def freeze_everything(self):
        for param in self.clip_model.parameters():
            param.requires_grad = False

    def encode_text(self, text, normalize: bool = True):
        print("encode_text")
        cast_dtype = self.clip_model.transformer.get_cast_dtype()

        x = self.clip_model.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.clip_model.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip_model.transformer(x, attn_mask=self.clip_model.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clip_model.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.clip_model.text_projection
        return F.normalize(x, dim=-1) if normalize else x
    def tokenize_text(self, text):
        return self.text_tokenizer(text)

    def extract_features(self, x):
        return {
            'convnext': self.extract_features_convnext,
            'resnet': self.extract_features_resnet,
        }[self.model_type](x)
    
    def visual_prediction_forward(self, x, masks=None):
        return {
            'convnext': self.visual_prediction_forward_convnext,
            'resnet': self.visual_prediction_forward_resnet,
        }[self.model_type](x, masks)

    def extract_features_convnext(self, x):
        out = {}
        x = self.clip_model.visual.trunk.stem(x)
        out['stem'] = x.contiguous() # os4
        for i in range(4):
            x = self.clip_model.visual.trunk.stages[i](x)
            out[f'res{i+2}'] = x.contiguous() # res 2 (os4), 3 (os8), 4 (os16), 5 (os32)
        
        x = self.clip_model.visual.trunk.norm_pre(x)
        out['clip_vis_dense'] = x.contiguous()
        return out
    
    def extract_features_resnet(self, x):
        out = {}
        x = self.clip_model.visual.act1(self.clip_model.visual.bn1(self.clip_model.visual.conv1(x)))
        x = self.clip_model.visual.act2(self.clip_model.visual.bn2(self.clip_model.visual.conv2(x)))
        x = self.clip_model.visual.act3(self.clip_model.visual.bn3(self.clip_model.visual.conv3(x)))
        out['stem'] = x.contiguous() # os2
        x = self.clip_model.visual.avgpool(x)
        x = self.clip_model.visual.layer1(x)
        out['res2'] = x.contiguous() # os4
        x = self.clip_model.visual.layer2(x)
        out['res3'] = x.contiguous() # os8
        x = self.clip_model.visual.layer3(x)
        out['res4'] = x.contiguous() # os16
        x = self.clip_model.visual.layer4(x)
        out['res5'] = x.contiguous() # os32
        out['clip_vis_dense'] = x
        return out

    def visual_prediction_forward_convnext(self, x, masks):
        batch, num_query, channel = x.shape
        x = x.reshape(batch*num_query, channel, 1, 1) # fake 2D input
        x = self.clip_model.visual.trunk.head(x)
        x = self.clip_model.visual.head(x)
        return x.view(batch, num_query, x.shape[-1]) # B x num_queries x 640

    def visual_prediction_forward_resnet(self, x, masks):
        batch, channel, height, width = x.shape
        if masks.shape[-2] != height or masks.shape[-1] != width:
            masks = F.inteprolate(masks, size=(height, width), mode='bilinear', align_corners=False)
        num_masks = masks.shape[1]

        positional_embedding = self.clip_model.visual.attnpool.positional_embedding.to(x.dtype)
        spatial_pos_embed = positional_embedding[1:, None, :] # HW x 1 x C
        orig_size = int(math.sqrt(spatial_pos_embed.shape[0]))
        spatial_pos_embed = spatial_pos_embed.permute(1, 2, 0).reshape(1, channel, orig_size, orig_size)
        spatial_pos_embed = F.interpolate(spatial_pos_embed, size=(height, width), mode='bilinear', align_corners=False) # 1 x C x H x W
        spatial_pos_embed = spatial_pos_embed.permute(2, 3, 0, 1).reshape(height*width, 1, channel)
        x = x.reshape(batch, channel, height * width).permute(2, 0, 1)  # BCHW -> (HW)BC
        key_value = x + spatial_pos_embed
        
        masks = masks.reshape(batch, num_masks, height * width)
        masks = (masks > 0).to(masks.dtype)
        query = x.mean(0, keepdim=True) + positional_embedding[:1, None, :]
        query = query.repeat_interleave(num_masks, dim=0)

        attn_mask = masks < 0.5
        attn_mask = attn_mask.unsqueeze(1).expand(-1, self.clip_model.visual.attnpool.num_heads, -1, -1)
        attn_mask = attn_mask.reshape(batch * self.clip_model.visual.attnpool.num_heads,
                                    query.shape[0], key_value.shape[0])

        x = F.multi_head_attention_forward(
            query=query, key=key_value, value=key_value,
            embed_dim_to_check=key_value.shape[-1],
            num_heads=self.clip_model.visual.attnpool.num_heads,
            q_proj_weight=self.clip_model.visual.attnpool.q_proj.weight,
            k_proj_weight=self.clip_model.visual.attnpool.k_proj.weight,
            v_proj_weight=self.clip_model.visual.attnpool.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.clip_model.visual.attnpool.q_proj.bias,
                                    self.clip_model.visual.attnpool.k_proj.bias,
                                    self.clip_model.visual.attnpool.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0.,
            out_proj_weight=self.clip_model.visual.attnpool.c_proj.weight,
            out_proj_bias=self.clip_model.visual.attnpool.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.clip_model.visual.attnpool.training,
            need_weights=False,
            attn_mask=attn_mask
        )[0].permute(1, 0, 2) # B x N x C

        return x

    def get_text_classifier(self, text_list, device):
        self.eval()
        with torch.no_grad():
            # reference for templates: https://github.com/mlfoundations/open_clip/blob/91f6cce16b7bee90b3b5d38ca305b5b3b67cc200/src/training/imagenet_zeroshot_data.py
            text_tokens = self.tokenize_text(text_list)
            text_tokens = text_tokens.to(device)
            # we return un-normalized text feature.
            text_features = self.encode_text(text_tokens, normalize=False)
            return text_features

    def forward(self, x):
        # print("asad")
        self.eval()
        with torch.no_grad():
            return self.extract_features(x)
    
    @property
    def dim_latent(self):
        return self.clip_model.text_projection.shape[-1]
    
    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in ["stem", "res2", "res3", "res4", "res5", "clip_embedding"]
        }

    @property
    def size_divisibility(self):
        return -1



import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import Conv2dNormActivation

class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积模块[4,5](@ref)"""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size, 
                                 stride=stride, padding=padding, groups=in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.norm = nn.GroupNorm(32, out_ch)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.norm(x)

class SEBlock(nn.Module):
    """轻量级通道注意力模块[1](@ref)"""
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.GELU(),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x + x * y.expand_as(x)

class FeatureProjector(nn.Module):
    def __init__(self, in_channels, out_dim=256):
        super().__init__()
        # 使用深度可分离卷积替代常规卷积[4,5](@ref)
        self.proj_res2 = DepthwiseSeparableConv(in_channels[0], out_dim)
        self.proj_res3 = DepthwiseSeparableConv(in_channels[1], out_dim)
        self.proj_res4 = DepthwiseSeparableConv(in_channels[2], out_dim)
        self.proj_res5 = DepthwiseSeparableConv(in_channels[3], out_dim)
        
    def forward(self, features):
        res2 = self.proj_res2(features["res2"])  # (B,256,H/4,W/4)
        res3 = self.proj_res3(features["res3"])  # (B,256,H/8,W/8)
        res4 = self.proj_res4(features["res4"])  # (B,256,H/16,W/16)
        res5 = self.proj_res5(features["res5"])  # (B,256,H/32,W/32)
        return [res2, res3, res4, res5]

class ProgressiveUpsampler(nn.Module):
    def __init__(self, in_dim=256):
        super().__init__()
        # 上采样卷积核组件（移除固定倍数上采样）
        self.up_conv5 = nn.Sequential(
            DepthwiseSeparableConv(in_dim, in_dim),
            # SEBlock(in_dim),
            # nn.GroupNorm(32, in_dim)
        )
        self.up_conv4 = nn.Sequential(
            DepthwiseSeparableConv(in_dim, in_dim),
            # SEBlock(in_dim),
            # nn.GroupNorm(32, in_dim)
        )
        self.up_conv3 = nn.Sequential(
            DepthwiseSeparableConv(in_dim, in_dim),
            # SEBlock(in_dim),
            # nn.GroupNorm(32, in_dim)
        )
        
        # 跳跃连接融合模块（保持不变）
        self.fuse_16 = nn.Sequential(
            DepthwiseSeparableConv(in_dim*2, in_dim),
            nn.GELU(),
            CoordAtt(in_dim, in_dim),
            nn.GroupNorm(32, in_dim)

        )
        self.fuse_8 = nn.Sequential(
            DepthwiseSeparableConv(in_dim*2, in_dim),
            nn.GELU(),
            CoordAtt(in_dim, in_dim),
            nn.GroupNorm(32, in_dim)
        )
        self.fuse_4 = nn.Sequential(
            DepthwiseSeparableConv(in_dim*2, in_dim),
            nn.GELU(),
            CoordAtt(in_dim, in_dim),
            nn.GroupNorm(32, in_dim)
        )
        
    def forward(self, features):
        f2, f3, f4, f5 = features
        
        # 第1步：f5→f4层级融合
        x = F.interpolate(f5, size=f4.shape[2:], mode='bilinear', align_corners=False)
        x = self.up_conv5(x)
        x = self.fuse_16(torch.cat([x, f4], dim=1))
        
        # 第2步：f4→f3层级融合 
        x = F.interpolate(x, size=f3.shape[2:], mode='bilinear', align_corners=False)
        x = self.up_conv4(x)
        x = self.fuse_8(torch.cat([x, f3], dim=1))
        
        # 第3步：f3→f2层级融合
        x = F.interpolate(x, size=f2.shape[2:], mode='bilinear', align_corners=False)
        x = self.up_conv3(x)
        x = self.fuse_4(torch.cat([x, f2], dim=1))
        
        return x

class MaskPredictor(nn.Module):
    def __init__(self, in_dim=256, num_classes=150):
        super().__init__()
        # 4倍上采样至原图尺寸[3](@ref)
        self.final_upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            DepthwiseSeparableConv(in_dim, in_dim),
        )
        self.linear1 = nn.Linear(in_dim, 128)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(128, 128)
        # self.conv_logits = nn.Conv2d(in_dim, num_classes, 1)
        
    def forward(self, x):
        x = self.linear2(self.act(self.linear1(self.final_upsample(x).permute(0,2,3,1))))  # (B,256,H,W)
        return  x.permute(0,3,1,2) # self.conv_logits(x)  # (B,C,H,W)

class EnhancedCLIPPixelDecoder(nn.Module):
    def __init__(self, clip_backbone, num_classes):
        super().__init__()
        # 获取CLIP主干特征通道数
        in_channels = [
            clip_backbone._out_feature_channels["res2"],
            clip_backbone._out_feature_channels["res3"],
            clip_backbone._out_feature_channels["res4"],
            clip_backbone._out_feature_channels["res5"]
        ]
        
        self.feature_projector = FeatureProjector(in_channels)
        self.upsampler = ProgressiveUpsampler()
        self.mask_head = MaskPredictor(num_classes=num_classes)
        
        # 初始化参数
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, features):
        # 特征投影：统一通道数
        proj_feats = self.feature_projector(features)
        
        # 多尺度特征融合与上采样
        fused_feat = self.upsampler(proj_feats)

        return self.mask_head(fused_feat)