"""GraphMamba module with corrected per-sample node-message pairing.

Compared with ``gps_mamba.py``, this version only changes the layout used to
construct each two-node graph.  For every sample b, the graph is now built as
``[node_b, message_b]`` rather than concatenating all target nodes before all
message nodes.
"""

import inspect
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Dropout, Linear, Sequential
from torch_geometric.nn import GCNConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj
from torch_geometric.utils import degree, sort_edge_index, to_dense_batch

from mamba_ssm import Mamba


def activation_resolver(act_name: str):
    act_dict = {
        "relu": nn.ReLU(),
        "leaky_relu": nn.LeakyReLU(),
        "gelu": nn.GELU(),
        "sigmoid": nn.Sigmoid(),
        "tanh": nn.Tanh(),
        "prelu": nn.PReLU(),
        "elu": nn.ELU(),
        "none": nn.Identity(),
        None: nn.Identity(),
    }
    act_name = act_name.lower() if act_name else "none"
    return act_dict.get(act_name, nn.Identity())


def normalization_resolver(norm_name: str, channels: int, **kwargs):
    normalizations = {
        "batch": lambda: nn.BatchNorm1d(channels, **kwargs),
        "layer": lambda: nn.LayerNorm(channels, **kwargs),
        "instance": lambda: nn.InstanceNorm1d(channels, **kwargs),
        "group": lambda: nn.GroupNorm(
            num_groups=kwargs.get("num_groups", 32), num_channels=channels
        ),
    }
    return normalizations.get(norm_name.lower(), lambda: nn.Identity())()


def permute_within_batch(x, batch):
    if x is None:
        raise ValueError("输入x不能为None")
    if x.size(0) == 0:
        return torch.tensor([], device=x.device, dtype=torch.long)
    if batch is None:
        return torch.randperm(x.size(0), device=x.device)

    batch = batch.squeeze()
    if batch.dim() != 1 or batch.size(0) != x.size(0):
        raise ValueError("batch维度或长度不匹配")

    batch = batch.to(torch.long)
    perm_indices = torch.zeros_like(batch)
    for b in torch.unique(batch):
        indices = torch.where(batch == b)[0]
        perm_indices[indices] = indices[
            torch.randperm(indices.size(0), device=x.device)
        ]
    return perm_indices


class PPM(nn.Module):
    """金字塔池化模块，提取多尺度空间特征。"""

    def __init__(self, chnn_in, rd_sc, dila):
        super().__init__()
        chnn_out = chnn_in // rd_sc
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        chnn_in,
                        chnn_out,
                        3,
                        padding=d,
                        dilation=d,
                        bias=False,
                    ),
                    nn.BatchNorm2d(chnn_out),
                    nn.ReLU(inplace=True),
                )
                for d in dila
            ]
        )

    def forward(self, x):
        return [conv(x) for conv in self.convs]


class SpatialToNode(nn.Module):
    """在空间特征与图节点向量之间进行投影。"""

    def __init__(self, in_channels: int, hidden_dim: int):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.inv_proj = nn.Linear(hidden_dim, in_channels)

    def project(self, x: Tensor) -> Tensor:
        pooled = self.gap(x).view(x.size(0), -1)
        return self.proj(pooled)

    def restore(self, node_feat: Tensor, spatial_residual: Tensor) -> Tensor:
        channel_weight = torch.sigmoid(self.inv_proj(node_feat))
        weight_map = channel_weight.view(channel_weight.size(0), -1, 1, 1)
        return spatial_residual * weight_map + spatial_residual


class GPSConv(nn.Module):
    """局部GCN消息传递与全局Mamba序列建模的并行组合。"""

    def __init__(
        self,
        channels: int,
        conv: Optional[MessagePassing],
        heads: int = 1,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        act: str = "relu",
        att_type: str = "mamba",
        order_by_degree: bool = False,
        shuffle_ind: int = 0,
        d_state: int = 16,
        d_conv: int = 4,
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Optional[str] = "batch",
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.channels = channels
        self.conv = conv
        self.heads = heads
        self.dropout = dropout
        self.att_type = att_type
        self.shuffle_ind = shuffle_ind
        self.order_by_degree = order_by_degree

        assert not (order_by_degree and shuffle_ind != 0)

        if att_type == "transformer":
            self.attn = nn.MultiheadAttention(
                channels, heads, dropout=attn_dropout, batch_first=True
            )
        elif att_type == "mamba":
            self.self_attn = Mamba(
                d_model=channels,
                d_state=d_state,
                d_conv=d_conv,
                expand=1,
            )

        self.mlp = Sequential(
            Linear(channels, channels * 2),
            activation_resolver(act, **(act_kwargs or {})),
            Dropout(dropout),
            Linear(channels * 2, channels),
            Dropout(dropout),
        )

        norm_kwargs = norm_kwargs or {}
        self.norm1 = normalization_resolver(norm, channels, **norm_kwargs)
        self.norm2 = normalization_resolver(norm, channels, **norm_kwargs)
        self.norm3 = normalization_resolver(norm, channels, **norm_kwargs)

        self.norm_with_batch = False
        if self.norm1 is not None:
            self.norm_with_batch = (
                "batch" in inspect.signature(self.norm1.forward).parameters
            )

    def _apply_norm(self, norm_layer, x, batch):
        if norm_layer is None:
            return x
        return norm_layer(x, batch=batch) if self.norm_with_batch else norm_layer(x)

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        batch: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        hs = []

        if self.conv is not None:
            h = self.conv(x, edge_index, **kwargs)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = h + x
            h = self._apply_norm(self.norm1, h, batch)
            hs.append(h)

        if self.att_type == "transformer":
            h, mask = to_dense_batch(x, batch)
            h, _ = self.attn(
                h,
                h,
                h,
                key_padding_mask=~mask,
                need_weights=False,
            )
            h = h[mask]
        elif self.att_type == "mamba":
            if self.order_by_degree:
                deg = degree(edge_index[0], x.shape[0]).to(torch.long)
                order_tensor = torch.stack([batch, deg], 1).T
                _, x = sort_edge_index(order_tensor, edge_attr=x)

            if self.shuffle_ind == 0:
                h, mask = to_dense_batch(x, batch)
                h = self.self_attn(h)[mask]
            else:
                mamba_arr = []
                for _ in range(self.shuffle_ind):
                    perm = permute_within_batch(x, batch)
                    h_i, mask = to_dense_batch(x[perm], batch)
                    h_i = self.self_attn(h_i)[mask][perm]
                    mamba_arr.append(h_i)
                h = sum(mamba_arr) / self.shuffle_ind

        h = F.dropout(h, p=self.dropout, training=self.training)
        h = h + x
        h = self._apply_norm(self.norm2, h, batch)
        hs.append(h)

        out = sum(hs)
        out = out + self.mlp(out)
        out = self._apply_norm(self.norm3, out, batch)
        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.channels}, "
            f"conv={self.conv}, heads={self.heads})"
        )


def _get_msg(x1, x2, C_wgt):
    """根据节点差异计算门控消息。"""

    b, c, h, w = x1.shape
    n_node = C_wgt.groups
    if c % n_node != 0:
        raise ValueError(f"通道数 {c} 必须能被节点数 {n_node} 整除")
    wgt = (
        C_wgt(x1 - x2)
        .unsqueeze(1)
        .repeat(1, c // n_node, 1, 1, 1)
        .view(b, c, h, w)
    )
    return x2 * torch.sigmoid(wgt)


class MGraphModel(nn.Module):
    """单模态图推理模型。"""

    def __init__(
        self,
        n_node,
        chnn_ppm,
        hidden_dim,
        heads=4,
        dropout=0.1,
        att_type="mamba",
    ):
        super().__init__()
        self.n_node = n_node
        self.chnn_ppm = chnn_ppm
        self.hidden_dim = hidden_dim

        self.spatial_to_node = SpatialToNode(
            in_channels=chnn_ppm,
            hidden_dim=hidden_dim,
        )
        self.C_wgt = nn.Conv2d(
            chnn_ppm * (n_node - 1),
            n_node - 1,
            1,
            groups=n_node - 1,
            bias=False,
        )
        self.gps_conv = GPSConv(
            channels=hidden_dim,
            conv=GCNConv(hidden_dim, hidden_dim),
            heads=heads,
            dropout=dropout,
            att_type=att_type,
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def _build_graph(self, node_features_2d):
        """为每个样本构建 ``目标节点 <-> 自身消息节点`` 的二节点图。"""

        total_nodes, _ = node_features_2d.shape
        if total_nodes % 2 != 0:
            raise ValueError(
                f"二节点图要求节点总数为偶数，当前为 {total_nodes}"
            )

        b = total_nodes // 2
        device = node_features_2d.device
        batch_idx = torch.arange(b, device=device).repeat_interleave(2)

        graph_start = torch.arange(b, device=device) * 2
        src = torch.stack([graph_start, graph_start + 1], dim=1).reshape(-1)
        dst = torch.stack([graph_start + 1, graph_start], dim=1).reshape(-1)
        edge_index = torch.stack([src, dst], dim=0)
        return node_features_2d, edge_index, batch_idx

    def forward(self, feat_list):
        b, c, h, w = feat_list[0].shape
        pred_s = []

        for idx in range(self.n_node):
            h_t = feat_list[idx]
            other_idx = [i for i in range(self.n_node) if i != idx]

            h_t_expand = h_t.repeat(1, len(other_idx), 1, 1)
            h_neighbors = torch.cat(
                [feat_list[i] for i in other_idx],
                dim=1,
            )
            msg = _get_msg(h_t_expand, h_neighbors, self.C_wgt)
            m_t = msg.view(b, len(other_idx), c, h, w).sum(dim=1)

            node_vec = self.spatial_to_node.project(h_t)
            msg_vec = self.spatial_to_node.project(m_t)

            # Pair-fix: [node0, msg0, node1, msg1, ...].
            # 因此每个二节点图只包含同一个样本的目标节点和消息节点。
            x_graph = torch.stack([node_vec, msg_vec], dim=1).reshape(
                b * 2, self.hidden_dim
            )
            x_graph, edge_index, batch_idx = self._build_graph(x_graph)

            x_updated = self.gps_conv(
                x_graph,
                edge_index,
                batch=batch_idx,
            )

            # Pair-fix: 目标节点位于每个二节点图的第0位，即全局偶数位置。
            node_updated = x_updated[0::2]

            h_t_updated = self.spatial_to_node.restore(node_updated, h_t)
            pred_s.append(h_t_updated * self.gamma + h_t)

        return pred_s


class MGraphReasoning(nn.Module):
    """双模态GraphMamba推理模块。"""

    def __init__(
        self,
        chnn_in,
        rd_sc,
        dila,
        n_iter,
        heads=4,
        dropout=0.1,
        att_type="mamba",
        hidden_dim=None,
    ):
        super().__init__()
        self.n_iter = n_iter
        self.n_node = len(dila)
        self.chnn_ppm = chnn_in // rd_sc

        self.ppm_ir = PPM(chnn_in, rd_sc, dila)
        self.ppm_vis = PPM(chnn_in, rd_sc, dila)

        if hidden_dim is None:
            hidden_dim = max(self.chnn_ppm * 4, 32)

        self.graph_ir = MGraphModel(
            self.n_node,
            self.chnn_ppm,
            hidden_dim,
            heads,
            dropout,
            att_type,
        )
        self.graph_vis = MGraphModel(
            self.n_node,
            self.chnn_ppm,
            hidden_dim,
            heads,
            dropout,
            att_type,
        )

        self.C_pa = nn.ModuleList(
            [nn.Conv2d(self.chnn_ppm, 1, 1, bias=False) for _ in range(2)]
        )

        chnn_ctx = self.chnn_ppm * 2
        self.C_ca = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(
                        chnn_ctx,
                        chnn_ctx // 4,
                        1,
                        bias=False,
                    ),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        chnn_ctx // 4,
                        self.chnn_ppm,
                        1,
                        bias=False,
                    ),
                )
                for _ in range(2)
            ]
        )

    def _int(self, feat_ir, feat_vis):
        gate_ir = torch.sigmoid(self.C_pa[0](feat_ir - feat_vis))
        gate_vis = torch.sigmoid(self.C_pa[1](feat_vis - feat_ir))
        out_vis = feat_ir * gate_ir + feat_vis
        out_ir = feat_vis * gate_vis + feat_ir
        return out_ir, out_vis

    def _enh(self, func, ctx, feat):
        weight = torch.sigmoid(func(ctx))
        return feat * weight + feat

    def forward(self, inputs, node=False):
        feat_ir, feat_vis, nd_ir, nd_vis = inputs

        feat_ir_list = self.ppm_ir(feat_ir)
        feat_vis_list = self.ppm_vis(feat_vis)

        if node:
            feat_ir_list = [
                self._enh(self.C_ca[0], nd_ir, f) for f in feat_ir_list
            ]
            feat_vis_list = [
                self._enh(self.C_ca[1], nd_vis, f) for f in feat_vis_list
            ]

        for i in range(self.n_node):
            feat_ir_list[i], feat_vis_list[i] = self._int(
                feat_ir_list[i], feat_vis_list[i]
            )
            feat_ir_list[i], feat_vis_list[i] = self._int(
                feat_ir_list[i], feat_vis_list[i]
            )

        feat_ir_list = self.graph_ir(feat_ir_list)
        feat_vis_list = self.graph_vis(feat_vis_list)
        return feat_ir_list, feat_vis_list
