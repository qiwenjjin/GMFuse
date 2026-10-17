import torch
import torch.nn as nn
import torch.nn.functional as F
from gps_mamba import MGraphReasoning


class SIM(nn.Module):
    # """
    # 语义调制模块 (Semantic Injection Module)。
    # 以语义先验图为条件，对特征图进行仿射变换调制。
    # 参考 SPADE 设计，先做 InstanceNorm 归一化，再用语义图生成 gamma/beta。
    # """
    def __init__(self, norm_nc, label_nc, nhidden=64):
        super().__init__()
        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.mlp_gamma = nn.Sequential(
            nn.Conv2d(nhidden, norm_nc, 3, padding=1),
            nn.Sigmoid()
        )
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, 3, padding=1)
        self.bn = nn.BatchNorm2d(norm_nc)

    def forward(self, x, segmap):
        normalized = self.param_free_norm(x)
        segmap = F.interpolate(segmap, size=x.shape[2:], mode='bilinear', align_corners=True)
        actv  = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta  = self.mlp_beta(actv)
        return self.bn(normalized * (1 + gamma)) + beta


class Encoder_ir(nn.Module):
    # """
    # 红外图像编码器。
    # 四层卷积逐步提取特征，最高层使用 MaxPool*AvgPool 的乘法融合，
    # 配合通道注意力和语义调制模块输出多尺度特征。
    # """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 128, 3, padding=1)
        self.conv2 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3 = nn.Conv2d(256, 384, 3, padding=1)
        self.conv4 = nn.Conv2d(384, 512, 3, padding=1)
        self.maxpool = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.avgpool = nn.AvgPool2d(2, stride=2, ceil_mode=True)

        # 通道注意力 (SE-like)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(512, 32, bias=False)
        self.fc2 = nn.Linear(32, 512, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.sim = SIM(norm_nc=512, label_nc=1, nhidden=64)

    def forward(self, x, semantic_prior=None):
        x1 = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        x1_pool = self.maxpool(x1)

        x2 = F.leaky_relu(self.conv2(x1_pool), negative_slope=0.2)
        x2_pool = self.maxpool(x2)

        x3 = F.leaky_relu(self.conv3(x2_pool), negative_slope=0.2)
        x4 = F.leaky_relu(self.conv4(x3), negative_slope=0.2)

        # 红外使用乘法融合两种池化，保留显著响应
        x_high = torch.mul(self.maxpool(x4), self.avgpool(x4))

        # 通道注意力
        w = self.gap(x_high).view(x_high.size(0), -1)
        w = self.sigmoid(self.fc2(self.relu(self.fc1(w))))
        x_high = x_high * w.view(x_high.size(0), -1, 1, 1)

        if semantic_prior is not None:
            x_high = self.sim(x_high, semantic_prior)

        return x_high, x2_pool, x1_pool   # L3, L2, L1


class Encoder_vis(nn.Module):
    # """
    # 可见光图像编码器。
    # 结构与红外编码器相同，最高层使用加法融合两种池化，
    # 加法融合保留更多纹理细节，与红外的乘法融合形成互补。
    # """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 128, 3, padding=1)
        self.conv2 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3 = nn.Conv2d(256, 384, 3, padding=1)
        self.conv4 = nn.Conv2d(384, 512, 3, padding=1)
        self.maxpool = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.avgpool = nn.AvgPool2d(2, stride=2, ceil_mode=True)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(512, 32, bias=False)
        self.fc2 = nn.Linear(32, 512, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.sim = SIM(norm_nc=512, label_nc=1, nhidden=64)

    def forward(self, x, semantic_prior=None):
        x1 = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        x1_pool = self.maxpool(x1)

        x2 = F.leaky_relu(self.conv2(x1_pool), negative_slope=0.2)
        x2_pool = self.maxpool(x2)

        x3 = F.leaky_relu(self.conv3(x2_pool), negative_slope=0.2)
        x4 = F.leaky_relu(self.conv4(x3), negative_slope=0.2)

        # 可见光使用加法融合两种池化，保留更多纹理细节
        x_high = torch.add(self.maxpool(x4), self.avgpool(x4))

        # 通道注意力
        w = self.gap(x_high).view(x_high.size(0), -1)
        w = self.sigmoid(self.fc2(self.relu(self.fc1(w))))
        x_high = x_high * w.view(x_high.size(0), -1, 1, 1)

        if semantic_prior is not None:
            x_high = self.sim(x_high, semantic_prior)

        return x_high, x2_pool, x1_pool   # L3, L2, L1


class CGR(nn.Module):
    # """
    # 跨模态图推理 + 渐进式融合解码器。

    # 编码器输出三个尺度的特征对 (L3/L2/L1)，
    # 每个尺度经 MGraphReasoning 进行双模态图推理后，
    # 自顶向下渐进融合，最终输出单通道融合图。

    # 通道流：
    #   L3: chnn_side[0]=512, C_ppm=16, fuse->128
    #   L2: chnn_side[1]=256, C_ppm=8,  decode->32
    #   L1: chnn_side[2]=128, C_ppm=4,  decode->4
    #   out: 4->1
    # """
    def __init__(
        self,
        chnn_side=(512, 256, 128),
        chnn_targ=(512, 128, 32, 4),
        rd_sc=32,
        dila=(4, 8, 16),
        hidden_dim_gnn=None
    ):
        super().__init__()
        self.n_graph = len(chnn_side)
        n_node = len(dila)

        # hidden_dim 策略：若未指定则由 MGraphReasoning 内部自适应
        # 若指定整数则所有层共用，若指定列表则按层分配
        if isinstance(hidden_dim_gnn, int):
            hidden_dims = [hidden_dim_gnn] * self.n_graph
        elif hidden_dim_gnn is None:
            hidden_dims = [None] * self.n_graph   # 交由 MGraphReasoning 自适应
        else:
            hidden_dims = hidden_dim_gnn          # 手动列表 e.g. [64, 32, 16]

        # 三个尺度的图推理模块
        self.graph = nn.ModuleList([
            MGraphReasoning(
                chnn_in=chnn_side[i],
                rd_sc=rd_sc,
                dila=dila,
                n_iter=2,
                att_type='mamba',
                hidden_dim=hidden_dims[i]
            ) for i in range(self.n_graph)
        ])

        # 每个尺度的 PPM 输出拼接卷积（IR 和 VIS 各一组）
        # 输入：n_node 个 PPM 特征拼接，输出：单个 C_ppm 特征
        self.C_cat_ir = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(chnn_side[i] // rd_sc * n_node, chnn_side[i] // rd_sc, 3, 1, 1, bias=False),
                nn.BatchNorm2d(chnn_side[i] // rd_sc),
                nn.ReLU(inplace=True)
            ) for i in range(self.n_graph)
        ])
        self.C_cat_vis = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(chnn_side[i] // rd_sc * n_node, chnn_side[i] // rd_sc, 3, 1, 1, bias=False),
                nn.BatchNorm2d(chnn_side[i] // rd_sc),
                nn.ReLU(inplace=True)
            ) for i in range(self.n_graph)
        ])

        # L3 跨模态融合层
        # 输入：IR C_ppm + VIS C_ppm = 16*2=32，输出：chnn_targ[1]=128
        chnn_L3_r = chnn_side[0] // rd_sc
        self.fuse_L3 = nn.Sequential(
            nn.Conv2d(chnn_L3_r * 2, chnn_targ[1], 1, bias=False),
            nn.BatchNorm2d(chnn_targ[1]),
            nn.ReLU(inplace=True)
        )

        # L2 解码层：上采样的 L3 融合特征 + L2 双模态特征 -> chnn_targ[2]
        chnn_L2_r = chnn_side[1] // rd_sc
        self.decode_L2 = self._make_decoder_block(
            chnn_in=chnn_targ[1] + chnn_L2_r * 2,
            chnn_out=chnn_targ[2]
        )

        # L1 解码层：上采样的 L2 融合特征 + L1 双模态特征 -> chnn_targ[3]
        chnn_L1_r = chnn_side[2] // rd_sc
        self.decode_L1 = self._make_decoder_block(
            chnn_in=chnn_targ[2] + chnn_L1_r * 2,
            chnn_out=chnn_targ[3]
        )

        # 最终输出卷积
        self.C_cls = nn.Conv2d(chnn_targ[3], 1, 1)

    @staticmethod
    def _make_decoder_block(chnn_in, chnn_out):
        return nn.Sequential(
            nn.Conv2d(chnn_in, chnn_out, 3, 1, 1, bias=False),
            nn.BatchNorm2d(chnn_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(chnn_out, chnn_out, 3, 1, 1, bias=False),
            nn.BatchNorm2d(chnn_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, inputs):
        # inputs[0] = (ir_L3, ir_L2, ir_L1)
        # inputs[1] = (vis_L3, vis_L2, vis_L1)
        img, depth = inputs

        nd_ir, nd_vis, nd_key = None, None, False

        # --- L3 (H/8, W/8) ---
        ir_feats_L3, vis_feats_L3 = self.graph[0](
            [img[0], depth[0], nd_ir, nd_vis], nd_key
        )
        ir_L3  = self.C_cat_ir[0](torch.cat(ir_feats_L3, dim=1))
        vis_L3 = self.C_cat_vis[0](torch.cat(vis_feats_L3, dim=1))
        fused_L3 = self.fuse_L3(torch.cat([ir_L3, vis_L3], dim=1))

        nd_ir, nd_vis, nd_key = ir_L3, vis_L3, True

        # --- L2 (H/4, W/4) ---
        ir_feats_L2, vis_feats_L2 = self.graph[1](
            [img[1], depth[1], nd_ir, nd_vis], nd_key
        )
        ir_L2  = self.C_cat_ir[1](torch.cat(ir_feats_L2, dim=1))
        vis_L2 = self.C_cat_vis[1](torch.cat(vis_feats_L2, dim=1))
        fused_L3_up = F.interpolate(fused_L3, scale_factor=2, mode='bilinear', align_corners=True)
        fused_L2 = self.decode_L2(torch.cat([fused_L3_up, ir_L2, vis_L2], dim=1))

        nd_ir, nd_vis = ir_L2, vis_L2

        # --- L1 (H/2, W/2) ---
        ir_feats_L1, vis_feats_L1 = self.graph[2](
            [img[2], depth[2], nd_ir, nd_vis], nd_key
        )
        ir_L1  = self.C_cat_ir[2](torch.cat(ir_feats_L1, dim=1))
        vis_L1 = self.C_cat_vis[2](torch.cat(vis_feats_L1, dim=1))
        fused_L2_up = F.interpolate(fused_L2, scale_factor=2, mode='bilinear', align_corners=True)
        fused_L1 = self.decode_L1(torch.cat([fused_L2_up, ir_L1, vis_L1], dim=1))

        # --- 输出 (H, W) ---
        out = self.C_cls(
            F.interpolate(fused_L1, scale_factor=2, mode='bilinear', align_corners=True)
        )
        return out


class backbone(nn.Module):
    # """
    # 完整融合网络。

    # 两阶段前向传播：
    #   Pass 1：仅用可见光编码器生成语义先验（单模态，轻量）
    #   Pass 2：将语义先验注入双模态编码器，再经图推理渐进解码输出融合图
    # """
    def __init__(self, num_seg_classes=21, rd_sc=32, hidden_dim_gnn=None):
        super().__init__()
        self.vis_conv = Encoder_vis()
        self.inf_conv = Encoder_ir()
        self.graph = CGR(
            chnn_side=(512, 256, 128),
            chnn_targ=(512, 128, 32, 4),
            rd_sc=rd_sc,
            dila=(4, 8, 16),
            hidden_dim_gnn=hidden_dim_gnn
        )
        # 语义先验头：仅基于可见光 L3 特征 (512通道)
        self.seg_head = nn.Sequential(
            nn.Conv2d(512, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_seg_classes, 1)
        )

    def _to_gray(self, image_vis):
        # """RGB 转灰度（ITU-R BT.601），单通道输入直接返回"""
        if image_vis.shape[1] == 3:
            return (0.299 * image_vis[:, 0:1] +
                    0.587 * image_vis[:, 1:2] +
                    0.114 * image_vis[:, 2:3])
        return image_vis

    def forward(self, image_vis, image_ir):
        image_vis_gray = self._to_gray(image_vis)

        # Pass 1：单模态语义先验生成（仅可见光）
        vis_feat_init, _, _ = self.vis_conv(image_vis_gray, semantic_prior=None)
        semantic_logits = self.seg_head(vis_feat_init)
        semantic_prior = torch.sigmoid(semantic_logits.mean(dim=1, keepdim=True))

        # Pass 2：语义先验注入双模态编码器
        vis_feats = self.vis_conv(image_vis_gray, semantic_prior)
        ir_feats  = self.inf_conv(image_ir, semantic_prior)

        # Pass 3：图推理 + 渐进式解码
        return self.graph([ir_feats, vis_feats])

