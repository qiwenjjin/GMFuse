import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class OhemCELoss(nn.Module):
    # """在线难例挖掘交叉熵损失，用于分割任务"""
    def __init__(self, thresh=0.7, n_min=1000, ignore_lb=255):
        super().__init__()
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float))
        self.n_min = n_min
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

    def forward(self, logits, labels):
        loss = self.criteria(logits, labels).view(-1)
        loss, _ = torch.sort(loss, descending=True)
        if loss[self.n_min] > self.thresh:
            loss = loss[loss > self.thresh]
        else:
            loss = loss[:self.n_min]
        return loss.mean()


class FusionLoss(nn.Module):
    # """
    # 红外与可见光融合损失。

    # 三项组成：
    #   pixel_loss:    自适应加权像素损失，根据局部对比度动态决定
    #                  融合图更偏向IR还是VIS
    #   edge_loss:     融合图的梯度应保留两个源中较强的边缘
    #   ssim_loss:     与两个源的结构相似性加权之和（IR权重0.6，VIS权重0.4）

    # 自适应权重的设计依据：
    #   - 红外图像在热目标区域对比度高 → 该区域融合图应更接近IR
    #   - 可见光在纹理丰富区域对比度高 → 该区域融合图应更接近VIS
    #   - 用局部方差衡量各模态的局部信息量，归一化后作为逐像素混合权重
    # """
    def __init__(self, alpha=10.0, beta=0.5):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.mse   = nn.MSELoss()

        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

        # 用于计算局部方差的均值核（5x5）
        avg_kernel = torch.ones(1, 1, 5, 5) / 25.0
        self.register_buffer('avg_kernel', avg_kernel)

    def _gradient(self, x):
        gx = F.conv2d(F.pad(x, (1,1,1,1), mode='replicate'), self.sobel_x)
        gy = F.conv2d(F.pad(x, (1,1,1,1), mode='replicate'), self.sobel_y)
        return torch.sqrt(gx**2 + gy**2 + 1e-6)

    def _local_variance(self, x):
        # """计算局部方差，衡量各位置的信息量"""
        mean = F.conv2d(F.pad(x, (2,2,2,2), mode='replicate'), self.avg_kernel)
        mean_sq = F.conv2d(F.pad(x**2, (2,2,2,2), mode='replicate'), self.avg_kernel)
        return (mean_sq - mean**2).clamp(min=0)

    def _adaptive_pixel_target(self, vis, ir):
        # """
        # 根据局部方差计算自适应混合目标：
        #   w_ir = var_ir / (var_ir + var_vis + eps)
        #   target = w_ir * ir + (1 - w_ir) * vis
        # 信息量更大的模态在该位置贡献更多。
        # """
        var_vis = self._local_variance(vis)
        var_ir  = self._local_variance(ir)
        w_ir = var_ir / (var_ir + var_vis + 1e-6)
        return w_ir * ir + (1 - w_ir) * vis

    def _ssim(self, x, y, window_size=11):
        C1, C2 = 0.01**2, 0.03**2
        coords = torch.arange(window_size, device=x.device).float() - window_size // 2
        gauss  = torch.exp(-coords**2 / (2 * 1.5**2))
        gauss  = gauss / gauss.sum()
        kernel = gauss.outer(gauss).view(1, 1, window_size, window_size)
        pad    = window_size // 2

        mu_x  = F.conv2d(x,   kernel, padding=pad)
        mu_y  = F.conv2d(y,   kernel, padding=pad)
        mu_xx = F.conv2d(x*x, kernel, padding=pad) - mu_x**2
        mu_yy = F.conv2d(y*y, kernel, padding=pad) - mu_y**2
        mu_xy = F.conv2d(x*y, kernel, padding=pad) - mu_x * mu_y

        ssim_map = ((2*mu_x*mu_y + C1) * (2*mu_xy + C2)) / \
                   ((mu_x**2 + mu_y**2 + C1) * (mu_xx + mu_yy + C2))
        return ssim_map.mean()

    def forward(self, fused, vis, ir):
        # 自适应像素目标（比硬性max更合理）
        pixel_target = self._adaptive_pixel_target(vis, ir)
        pixel_loss   = self.mse(fused, pixel_target)

        # 边缘损失：保留两个源中较强的边缘
        edge_loss = F.l1_loss(
            self._gradient(fused),
            torch.max(self._gradient(vis), self._gradient(ir))
        )

        # SSIM损失：IR权重略高（热目标结构更重要）
        ssim_loss = (1 - self._ssim(fused, vis)) * 0.4 + \
                    (1 - self._ssim(fused, ir))  * 0.6

        return pixel_loss + self.alpha * edge_loss + self.beta * ssim_loss


class FusionLossWithDetails(FusionLoss):
    # """返回各分项损失值，用于TensorBoard监控"""
    def forward(self, fused, vis, ir):
        pixel_target = self._adaptive_pixel_target(vis, ir)
        pixel_loss   = self.mse(fused, pixel_target)

        edge_loss = F.l1_loss(
            self._gradient(fused),
            torch.max(self._gradient(vis), self._gradient(ir))
        )

        ssim_vis  = 1 - self._ssim(fused, vis)
        ssim_ir   = 1 - self._ssim(fused, ir)
        ssim_loss = ssim_vis * 0.4 + ssim_ir * 0.6

        total = pixel_loss + self.alpha * edge_loss + self.beta * ssim_loss

        details = {
            'pixel':    pixel_loss,
            'edge':     edge_loss,
            'ssim':     ssim_loss,
            'ssim_vis': ssim_vis,
            'ssim_ir':  ssim_ir,
        }
        return total, details
