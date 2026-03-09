from functools import partial
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
import spconv.pytorch as spconv
from timm.layers import trunc_normal_
from torch_geometric.nn.pool import voxel_grid
from torch_geometric.utils import scatter
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                             padding=1, bias=False, indice_key=indice_key)


def conv1x3(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(1, 3, 3), stride=stride,
                             padding=(0, 1, 1), bias=False, indice_key=indice_key)


def conv1x1x3(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(1, 1, 3), stride=stride,
                             padding=(0, 0, 1), bias=False, indice_key=indice_key)


def conv1x3x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(1, 3, 1), stride=stride,
                             padding=(0, 1, 0), bias=False, indice_key=indice_key)


def conv3x1x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(3, 1, 1), stride=stride,
                             padding=(1, 0, 0), bias=False, indice_key=indice_key)


def conv3x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(3, 1, 3), stride=stride,
                             padding=(1, 0, 1), bias=False, indice_key=indice_key)


def conv1x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=1, stride=stride,
                             padding=1, bias=False, indice_key=indice_key)

class ResContextBlock(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size=(3, 3, 3), stride=1, indice_key=None):
        super(ResContextBlock, self).__init__()
        self.conv1 = conv1x3(in_filters, out_filters, indice_key=indice_key + "bef1")
        self.bn0 = nn.BatchNorm1d(out_filters)
        self.act1 = nn.LeakyReLU()

        self.conv1_2 = conv3x1(out_filters, out_filters, indice_key=indice_key + "bef2")
        # self.conv1_2 = conv1x3(out_filters, out_filters, indice_key=indice_key + "bef")

        self.bn0_2 = nn.BatchNorm1d(out_filters)
        self.act1_2 = nn.LeakyReLU()

        self.conv2 = conv3x1(in_filters, out_filters, indice_key=indice_key + "bef3")
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(out_filters)

        self.conv3 = conv1x3(out_filters, out_filters, indice_key=indice_key + "bef4")
        # self.conv3 = conv3x1(out_filters, out_filters, indice_key=indice_key + "bef")
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm1d(out_filters)

        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut = shortcut.replace_feature(self.act1(shortcut.features))
        shortcut = shortcut.replace_feature(self.bn0(shortcut.features))

        shortcut = self.conv1_2(shortcut)
        shortcut = shortcut.replace_feature(self.act1_2(shortcut.features))
        shortcut = shortcut.replace_feature(self.bn0_2(shortcut.features))

        resA = self.conv2(x)
        resA = resA.replace_feature(self.act2(resA.features))
        resA = resA.replace_feature(self.bn1(resA.features))

        resA = self.conv3(resA)
        resA = resA.replace_feature(self.act3(resA.features))
        resA = resA.replace_feature(self.bn2(resA.features))
        resA = resA.replace_feature(resA.features + shortcut.features)

        return resA


class BasicBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        embed_channels,
        norm_fn=None,
        indice_key=None,
        depth=4,
        groups=None,
        grid_size=None,
        bias=False,
    ):
        super().__init__()
        assert embed_channels % groups == 0
        self.groups = groups
        self.embed_channels = embed_channels
        self.proj = nn.ModuleList()
        self.grid_size = grid_size
        self.weight = nn.ModuleList()
        self.l_w = nn.ModuleList()
        self.proj.append(
            nn.Sequential(
                nn.Linear(embed_channels, embed_channels, bias=False),
                norm_fn(embed_channels),
                nn.ReLU(),
            )
        )
        for _ in range(depth - 1):
            self.proj.append(
                nn.Sequential(
                    nn.Linear(embed_channels, embed_channels, bias=False),
                    norm_fn(embed_channels),
                    nn.ReLU(),
                )
            )
            self.l_w.append(
                nn.Sequential(
                    nn.Linear(embed_channels, embed_channels, bias=False),
                    norm_fn(embed_channels),
                    nn.ReLU(),
                )
            )
            self.weight.append(nn.Linear(embed_channels, embed_channels, bias=False))

        self.adaptive = nn.Linear(embed_channels, depth - 1, bias=False)
        self.fuse = nn.Sequential(
            nn.Linear(embed_channels * 2, embed_channels, bias=False),
            norm_fn(embed_channels),
            nn.ReLU(),
        )
        self.voxel_block = spconv.SparseSequential(
            spconv.SubMConv3d(
                embed_channels,
                embed_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                indice_key=indice_key,
                bias=bias,
            ),
            norm_fn(embed_channels),
            nn.ReLU(),
            spconv.SubMConv3d(
                embed_channels,
                embed_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                indice_key=indice_key,
                bias=bias,
            ),
            norm_fn(embed_channels),
        )
        self.act = nn.ReLU()

    def forward(self, x, clusters):
        feat = x.features
        feats = []
        for i, cluster in enumerate(clusters):
            pw = self.l_w[i](feat)
            pw = pw - scatter(pw, cluster, reduce="mean")[cluster]
            pw = self.weight[i](pw)
            pw = torch.exp(pw - pw.max())
            pw = pw / (scatter(pw, cluster, reduce="sum", dim=0)[cluster] + 1e-6)
            pfeat = self.proj[i](feat) * pw
            pfeat = scatter(pfeat, cluster, reduce="sum")[cluster]
            feats.append(pfeat)
        adp = self.adaptive(feat)
        adp = torch.softmax(adp, dim=1)
        feats = torch.stack(feats, dim=1)
        feats = torch.einsum("l n, l n c -> l c", adp, feats)
        feat = self.proj[-1](feat)
        feat = torch.cat([feat, feats], dim=1)
        feat = self.fuse(feat) + x.features
        res = feat
        x = x.replace_feature(feat)
        x = self.voxel_block(x)
        x = x.replace_feature(self.act(x.features + res))
        return x


class DonwBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        embed_channels,
        depth,
        sp_indice_key,
        point_grid_size,
        num_ref=16,
        groups=None,
        norm_fn=None,
        sub_indice_key=None,
        height_pooling=False,
    ):
        super().__init__()
        self.num_ref = num_ref
        self.depth = depth
        self.point_grid_size = point_grid_size
        self.height_pooling=height_pooling
        if self.height_pooling:
            self.pool = spconv.SparseSequential(
                spconv.SparseConv3d(
                    embed_channels,
                    embed_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    indice_key=sp_indice_key,
                    bias=False,
                ),
                norm_fn(embed_channels),
                nn.ReLU(),
            )
        else:
            self.pool = spconv.SparseSequential(
                spconv.SparseConv3d(
                    embed_channels,
                    embed_channels,
                    kernel_size=3,
                    stride=(2,2,1),
                    padding=1,
                    indice_key=sp_indice_key,
                    bias=False,
                ),
                norm_fn(embed_channels),
                nn.ReLU(),
            )
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(
                BasicBlock(
                    in_channels=in_channels,
                    embed_channels=in_channels,
                    depth=len(point_grid_size) + 1,
                    groups=groups,
                    grid_size=point_grid_size,
                    norm_fn=norm_fn,
                    indice_key=sub_indice_key,
                )
            )
        self.oalinear = nn.Sequential(
            nn.Linear(in_channels, embed_channels),
            nn.BatchNorm1d(embed_channels),
            nn.ReLU(),
        )
    def forward(self, x):
        coord = x.indices[:, 1:].float()
        batch = x.indices[:, 0]
        clusters = []
        for grid_size in self.point_grid_size:
            cluster = voxel_grid(pos=coord, size=grid_size, batch=batch)
            _, cluster = torch.unique(cluster, return_inverse=True)
            clusters.append(cluster)
        for block in self.blocks:
            x = block(x, clusters)
        x = x.replace_feature(self.oalinear(x.features))
        resA = self.pool(x)
        return resA, x


class ReconBlock(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size=(3, 3, 3), stride=1, indice_key=None):
        super(ReconBlock, self).__init__()
        self.conv1 = conv3x1x1(in_filters, out_filters, indice_key=indice_key + "bef1")
        self.bn0 = nn.BatchNorm1d(out_filters)
        self.act1 = nn.Sigmoid()

        self.conv1_2 = conv1x3x1(in_filters, out_filters, indice_key=indice_key + "bef2")
        self.bn0_2 = nn.BatchNorm1d(out_filters)
        self.act1_2 = nn.Sigmoid()

        self.conv1_3 = conv1x1x3(in_filters, out_filters, indice_key=indice_key + "bef3")
        self.bn0_3 = nn.BatchNorm1d(out_filters)
        self.act1_3 = nn.Sigmoid()

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut = shortcut.replace_feature(self.bn0(shortcut.features))
        shortcut = shortcut.replace_feature(self.act1(shortcut.features))

        shortcut2 = self.conv1_2(x)
        shortcut2 = shortcut2.replace_feature(self.bn0_2(shortcut2.features))
        shortcut2 = shortcut2.replace_feature(self.act1_2(shortcut2.features))

        shortcut3 = self.conv1_3(x)
        shortcut3 = shortcut.replace_feature(self.bn0_3(shortcut3.features))
        shortcut3 = shortcut3.replace_feature(self.act1_3(shortcut3.features))
        shortcut = shortcut.replace_feature(shortcut.features + shortcut2.features + shortcut3.features)

        shortcut = shortcut.replace_feature(shortcut.features * x.features)

        return shortcut


class UpBlock(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size=(3, 3, 3), indice_key=None, up_key=None):
        super(UpBlock, self).__init__()
        # self.drop_out = drop_out
        self.trans_dilao = conv3x3(in_filters, out_filters, indice_key=indice_key + "new_up")
        self.trans_act = nn.LeakyReLU()
        self.trans_bn = nn.BatchNorm1d(out_filters)

        self.conv1 = conv1x3(out_filters, out_filters, indice_key=indice_key + 'up1')
        self.act1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(out_filters)

        self.conv2 = conv3x1(out_filters, out_filters, indice_key=indice_key + 'up2')
        self.act2 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm1d(out_filters)

        self.conv3 = conv3x3(out_filters, out_filters, indice_key=indice_key + 'up3')
        self.act3 = nn.LeakyReLU()
        self.bn3 = nn.BatchNorm1d(out_filters)
        # self.dropout3 = nn.Dropout3d(p=dropout_rate)

        self.up_subm = spconv.SparseInverseConv3d(out_filters, out_filters, kernel_size=3, indice_key=up_key,
                                                  bias=False)

        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, skip):
        upA = self.trans_dilao(x)
        upA = upA.replace_feature(self.trans_act(upA.features))
        upA = upA.replace_feature(self.trans_bn(upA.features))

        ## upsample
        upA = self.up_subm(upA)

        upA = upA.replace_feature(upA.features + skip.features)

        upE = self.conv1(upA)
        upE = upE.replace_feature(self.act1(upE.features))
        upE = upE.replace_feature(self.bn1(upE.features))

        upE = self.conv2(upE)
        upE = upE.replace_feature(self.act2(upE.features))
        upE = upE.replace_feature(self.bn2(upE.features))

        upE = self.conv3(upE)
        upE = upE.replace_feature(self.act3(upE.features))
        upE = upE.replace_feature(self.bn3(upE.features))

        return upE

class OACNNs(nn.Module):
    def __init__(
        self,
        in_channels,
        num_classes,
        output_shape,
        embed_channels=64,
        enc_num_ref=[16, 16, 16, 16],
        enc_channels=[64, 64, 128, 256],
        groups=[2, 4, 8, 16],
        enc_depth=[3, 3, 9, 8],
        down_ratio=[2, 2, 2, 2],
        dec_channels=[64, 64, 128, 256],
        point_grid_size=[[16, 32, 64], [8, 16, 24], [4, 8, 12], [4, 8, 12]],
        dec_depth=[2, 2, 2, 2],
        height_pool = [True, True, False, False]
    ):
        super().__init__()
        sparse_shape = np.array(output_shape)
        # sparse_shape[0] = 11
        print(sparse_shape)
        self.sparse_shape = sparse_shape

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_stages = len(enc_channels)
        self.embed_channels = embed_channels
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.downCntx = ResContextBlock(in_channels, embed_channels, indice_key="pre")
       
        self.enc = nn.ModuleList()
        self.dec = nn.ModuleList()
        for i in range(self.num_stages):
            self.enc.append(
                DonwBlock(
                    in_channels=embed_channels if i == 0 else enc_channels[i - 1],
                    embed_channels=enc_channels[i],
                    depth=enc_depth[i],
                    norm_fn=norm_fn,
                    groups=groups[i],
                    point_grid_size=point_grid_size[i],
                    num_ref=enc_num_ref[i],
                    sp_indice_key=f"spconv{i}",
                    sub_indice_key=f"subm{i + 1}",
                    height_pooling=height_pool[i]
                )
            )
            self.dec.append(
                UpBlock(
                    in_filters=(
                        enc_channels[-1]
                        if i == self.num_stages - 1
                        else dec_channels[i + 1]
                    ),
                    # skip_channels=embed_channels if i == 0 else enc_channels[i - 1],
                    out_filters=dec_channels[i],
                    # depth=dec_depth[i],
                    # norm_fn=norm_fn,
                    indice_key=f"subm{i}",
                    up_key=f"spconv{i}",
                )
            )
        self.upBlock_ins_heatmap = UpBlock(
                in_filters=(
                    dec_channels[1]
                ),
                # skip_channels=embed_channels,
                out_filters=dec_channels[0],
                # depth=dec_depth[0],
                # norm_fn=norm_fn,
                up_key=f"spconv{0}",
                indice_key="up3_ins_heatmap",
            )
        self.upBlock_ins_offset = UpBlock(
                in_filters=(
                    dec_channels[1]
                ),
                # skip_channels=embed_channels,
                out_filters=dec_channels[0],
                # depth=dec_depth[0],
                # norm_fn=norm_fn,
                up_key=f"spconv{0}",
                indice_key="up3_ins_offset",
            )
        self.upBlock_ins_instmap = UpBlock(
                in_filters=(
                    dec_channels[1]
                ),
                # skip_channels=embed_channels,
                out_filters=dec_channels[0],
                # depth=dec_depth[0],
                # norm_fn=norm_fn,
                up_key=f"spconv{0}",
                indice_key="up3_ins_instmap",
            )
            

        self.logits = spconv.SubMConv3d(dec_channels[0], num_classes, kernel_size=3, stride=1, padding=1,bias=True)
        self.apply(self._init_weights)
        self.compress_offset = spconv.SubMConv3d(128 , 32, indice_key="compress_heatmap", kernel_size=3,
                                                 stride=1, padding=1,bias=True)
        self.compress_instmap = spconv.SubMConv3d(128, 32, indice_key="compress_instmap", kernel_size=3,
                                                 stride=1, padding=1,bias=True)

        self.pool3d_heatmap = spconv.SparseMaxPool3d((1, 1, 32), indice_key="pool3d_heatmap")
        self.pool3d_offset = spconv.SparseMaxPool3d((1, 1, 32), indice_key="pool3d_offset")
        self.pool3d_instmap = spconv.SparseMaxPool3d((1, 1, 32), indice_key="pool3d_instmap")
        self.logits_offset = nn.Conv2d(32, 2, 3, padding=(1, 0))
        self.logits_instmap = nn.Conv2d(32, 2, 3, padding=(1, 0))
        self.ReconNet_ins_heatmap = ReconBlock(64, 64, indice_key="recon_ins_heatmap")
        self.ReconNet_ins_offset = ReconBlock(64, 64, indice_key="recon_ins_offset")
        self.ReconNet_ins_instmap = ReconBlock(64, 64, indice_key="recon_ins_instmap")
        self.ReconNet = ReconBlock(64, 64, indice_key="recon")
        


    def forward(self, voxel_features, coors, batch_size,):
        coors = coors.int()
        x = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape,
                                    batch_size)
        x = self.downCntx(x)
        skips = []
        res = []
        for i in range(self.num_stages):
            x, resA = self.enc[i](x)
            skips.append(x)
            res.append(resA)
        x = skips.pop(-1)
        for i in reversed(range(self.num_stages)):
            skip = res.pop(-1)
            if i == 0:
                up1e_ins_heatmap = self.upBlock_ins_heatmap(x, skip)
                up0e_ins_heatmap = self.ReconNet_ins_heatmap(up1e_ins_heatmap)
                up0e_ins_heatmap = up0e_ins_heatmap.replace_feature(
                    torch.cat((up0e_ins_heatmap.features, up1e_ins_heatmap.features), 1))
                up0e_ins_heatmap = self.pool3d_heatmap(up0e_ins_heatmap)
                heatmap = up0e_ins_heatmap.dense().squeeze(-1)
                
                up1e_ins_offset = self.upBlock_ins_offset(x, skip)
                up0e_ins_offset = self.ReconNet_ins_offset(up1e_ins_offset)
                up0e_ins_offset = up0e_ins_offset.replace_feature(
                    torch.cat((up0e_ins_offset.features, up1e_ins_offset.features), 1))
                up0e_ins_offset = self.pool3d_offset(up0e_ins_offset)
                up0e_ins_offset = self.compress_offset(up0e_ins_offset)
                offset = up0e_ins_offset.dense().squeeze(-1)
                offset = F.pad(offset, (1, 1, 0, 0), mode='circular')
                offset = self.logits_offset(offset)

                up1e_ins_instmap = self.upBlock_ins_instmap(x, skip)
                up0e_ins_instmap = self.ReconNet_ins_instmap(up1e_ins_instmap)
                up0e_ins_instmap = up0e_ins_instmap.replace_feature(
                    torch.cat((up0e_ins_instmap.features, up1e_ins_instmap.features), 1))
                up0e_ins_instmap = self.pool3d_instmap(up0e_ins_instmap)
                up0e_ins_instmap = self.compress_instmap(up0e_ins_instmap)
                instmap = up0e_ins_instmap.dense().squeeze(-1)
                instmap = F.pad(instmap, (1, 1, 0, 0), mode='circular')
                instmap = self.logits_instmap(instmap)
                
                x = self.dec[i](x, skip)
                up0e = self.ReconNet(x)
                up0e = up0e.replace_feature(torch.cat((up0e.features, x.features), 1))
                break
            x = self.dec[i](x, skip)
        
        logits = self.logits(x)
        logits = logits.dense()
        return logits, heatmap, offset, instmap

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, spconv.SubMConv3d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

