# -*- coding: utf-8 -*-
# @Author : youngx
# @Time : 16:37  2022-05-06
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from einops import repeat, rearrange
import torch.nn.functional as F


class Patches(nn.Module):
    def __init__(self, pathsize, dim, channels):
        super(Patches, self).__init__()
        self.patchsize = (pathsize, pathsize)
        self.chans = pathsize * pathsize * channels
        self.Unfold = nn.Sequential(
            nn.Unfold(kernel_size=self.patchsize, stride=self.patchsize),
            Rearrange('b dim lenth-> b lenth dim')
        )
        self.proj = nn.Linear(self.chans, dim)

        # 直接使用卷积实现对输入图像patch 处理
        # self.proj = nn.Conv2d(3, dim, kernel_size=self.patchsize, stride=self.patchsize)

    def forward(self, x):
        x = self.Unfold(x)
        x = self.proj(x)  # .flatten(2).transpose(1, 2)
        return x


class FeadForward(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super(FeadForward, self).__init__()
        self.hidden_dim = dim // 4
        self.net = nn.Sequential(
            nn.Linear(dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, dim),
            nn.Dropout(dropout)
        )
        self.Norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x):
        shorrcut = x
        out = self.net(x)
        return self.Norm(shorrcut + out)


class MultiHeadAtttention(nn.Module):
    def __init__(self, dim, headNum=4, dim_head=64):
        super(MultiHeadAtttention, self).__init__()
        # self.inner_dim = headNum * dim_head
        self.inner_dim = dim_head // headNum
        self.headNum = headNum
        self.scale = self.inner_dim ** -0.5

        self.drop = nn.Dropout(0.2)
        self.qkv = nn.Sequential(
            nn.Linear(dim, dim_head * 3),
            Rearrange('b l (c num) -> b l c num', num=3),
        )
        self.rearange = Rearrange('b lenth (headNum dim)-> b headNum lenth dim', headNum=self.headNum)

        self.Norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x):
        shorcut = x
        qkv = self.qkv(x)
        q, k, v = qkv[..., 0], qkv[..., 1], qkv[..., 2]
        q = self.rearange(q)
        k = self.rearange(k)
        v = self.rearange(v)

        dot = q @ k.transpose(-2, -1) * self.scale  # [B, headNum, lenth, lenth]
        scores = self.drop(F.softmax(dot, dim=-1))
        scores = (scores @ v)  # [B, headNum, lenth, dim_head]
        out = rearrange(scores, 'b h n d -> b n (h d)')
        return self.Norm(out + shorcut)


class TransFormer(nn.Module):
    def __init__(self, layerNum, dim, multiheads, dim_head):
        super(TransFormer, self).__init__()
        self.layerNum = layerNum
        self.multiheads = multiheads
        self.layers = []
        for _ in range(self.layerNum):
            self.layers.append(MultiHeadAtttention(dim, self.multiheads, dim_head))
            self.layers.append(FeadForward(dim))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.layers(x)
        return x


class VIT(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, transLayer,
                 multiheads, pool='cls', channels=3, dropout=0.2):
        super(VIT, self).__init__()
        self.pool = pool
        num_patches = (image_size // patch_size) * (image_size // patch_size)
        self.inputEmbedding = Patches(pathsize=patch_size, dim=dim, channels=channels)
        self.positionEncoding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(dropout)

        self.transformer = TransFormer(transLayer, dim, multiheads, dim)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        x = self.inputEmbedding(x)
        b, n, _ = x.shape

        cls_tokens = self.cls_token.repeat(b, 1, 1)
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.positionEncoding
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        x = self.mlp_head(x)
        return x


# 测试 einops 中Rearrange, rearrange 可以在onnx中识别和使用
class VIT2(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, transLayer,
                 multiheads, pool='cls', channels=3, dropout=0.2):
        super(VIT2, self).__init__()
        self.rearrange = Rearrange('b h n d -> b n (h d)')

    def forward(self, x):
        x = self.rearrange(x)
        out = rearrange(x, 'b h w -> b w h')
        # x = torch.cat([out, out], dim=1)
        cls_tokens = repeat(out, 'b lenth dim -> b lenth dim d', d=1)
        return cls_tokens


if __name__ == '__main__':
    net = VIT(image_size=224, patch_size=16, num_classes=10, dim=256, transLayer=6, multiheads=8)
    img = torch.randn(2, 3, 224, 224)
    out = net(img)
    print(out.shape)
    net.eval()
    torch.onnx.export(net, img, "vit.onnx", verbose=1, training=torch.onnx.TrainingMode.EVAL,
                      input_names=["inputNode"], output_names=["outNode1"], opset_version=11, )
