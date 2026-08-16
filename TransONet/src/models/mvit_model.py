# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved. All Rights Reserved.


"""MViT models."""

import math
from functools import partial

import torch
import torch.nn as nn
from .attention import MultiScaleBlock
from .common import round_width
#from .misc import validate_checkpoint_wrapper_import
from torch.nn.init import trunc_normal_

from .build import MODEL_REGISTRY

try:
    from fairscale.nn.checkpoint import checkpoint_wrapper
except ImportError:
    checkpoint_wrapper = None


class PatchEmbed(nn.Module):
    """
    PatchEmbed.
    """

    def __init__(
        self,
        dim_in=128,
        dim_out=768,
        kernel=(7, 7),
        stride=(4, 4),
        padding=(3, 3),
    ):
        super().__init__()

        self.proj = nn.Conv2d(
            dim_in,
            dim_out,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
        )

    def forward(self, x):
        #print("x in trans type:"+str(x.type()))
        x = self.proj(x)
        # B C H W -> B HW C
        return x.transpose(1,3), x.shape #.flatten(2).transpose(1, 2)


class TransformerBasicHead(nn.Module):
    """
    Basic Transformer Head. No pool.
    """

    def __init__(
        self,
        dim_in,
        num_classes,
        dropout_rate=0.0,
        act_func="softmax",
    ):
        """
        Perform linear projection and activation as head for tranformers.
        Args:
            dim_in (int): the channel dimension of the input to the head.
            num_classes (int): the channel dimensions of the output to the head.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
        """
        super(TransformerBasicHead, self).__init__()
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        self.projection = nn.Linear(dim_in, num_classes, bias=True)
        #self.projection = nn.Dropout(dropout_rate)

        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=1)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation" "function.".format(act_func)
            )

    def forward(self, x):
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        x = self.projection(x)

        if not self.training:
            x = self.act(x)
        return x


@MODEL_REGISTRY.register()
class MViT(nn.Module):
    """
    Improved Multiscale Vision Transformers for Classification and Detection
    Yanghao Li*, Chao-Yuan Wu*, Haoqi Fan, Karttikeya Mangalam, Bo Xiong, Jitendra Malik,
        Christoph Feichtenhofer*
    https://arxiv.org/abs/2112.01526

    Multiscale Vision Transformers
    Haoqi Fan*, Bo Xiong*, Karttikeya Mangalam*, Yanghao Li*, Zhicheng Yan, Jitendra Malik,
        Christoph Feichtenhofer*
    https://arxiv.org/abs/2104.11227
    """

    def __init__(self):#, in_chans):#, cfg):
        super().__init__()
        # Get parameters.
        #assert cfg.DATA.TRAIN_CROP_SIZE == cfg.DATA.TEST_CROP_SIZE
        # Prepare input.
        in_chans = 128 #3
        spatial_size = 224
        # Prepare output.
        num_classes = in_chans #128 #256#384#13 #1000
        embed_dim = in_chans*2 #256 #512#768
        # MViT params.
        num_heads = 1
        depth = 3
        self.cls_embed_on = False
        self.use_abs_pos = False
        self.zero_decay_pos_cls = False

        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        # if cfg.MODEL.ACT_CHECKPOINT:
        #     validate_checkpoint_wrapper_import(checkpoint_wrapper)

        patch_embed = PatchEmbed(
            dim_in=in_chans,
            dim_out=embed_dim,
            kernel= [3,3],#[7,7],
            stride=[1,1],#[4, 4],
            padding=[1,1],#[3, 3],
        )
        # if cfg.MODEL.ACT_CHECKPOINT:
        #     patch_embed = checkpoint_wrapper(patch_embed)
        self.patch_embed = patch_embed

        patch_dims = [
            spatial_size // 4,
            spatial_size // 4,
        ]
        num_patches = math.prod(patch_dims)

        dpr = [
            x.item() for x in torch.linspace(0, 0.1, depth)
        ]  # stochastic depth decay rule

        if self.cls_embed_on:
            #print("zeros shape:"+ str(torch.zeros(1, 1, embed_dim).shape))
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            pos_embed_dim = num_patches + 1
        else:
            pos_embed_dim = num_patches

        if self.use_abs_pos:
            self.pos_embed = nn.Parameter(torch.zeros(1, pos_embed_dim, embed_dim))

        # MViT backbone configs
        dim_mul, head_mul, pool_q, pool_kv, stride_q, stride_kv = _prepare_mvit_configs()
        #     cfg
        # )

        input_size = patch_dims
        self.blocks = nn.ModuleList()
        for i in range(depth):
            num_heads = round_width(num_heads, head_mul[i])
            #if cfg.MVIT.DIM_MUL_IN_ATT:
            dim_out = round_width(
                embed_dim,
                dim_mul[i],
                divisor=round_width(num_heads, head_mul[i]),
            )
            # else:
            #     dim_out = round_width(
            #         embed_dim,
            #         dim_mul[i + 1],
            #         divisor=round_width(num_heads, head_mul[i + 1]),
            #     )
            attention_block = MultiScaleBlock(
                dim=embed_dim,
                dim_out=dim_out,
                num_heads=num_heads,
                input_size=input_size,
                mlp_ratio=4.0,
                qkv_bias=True,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                kernel_q=pool_q[i] if len(pool_q) > i else [],
                kernel_kv=pool_kv[i] if len(pool_kv) > i else [],
                stride_q=stride_q[i] if len(stride_q) > i else [],
                stride_kv=stride_kv[i] if len(stride_kv) > i else [],
                mode="conv",
                has_cls_embed=self.cls_embed_on,
                pool_first=False,
                rel_pos_spatial=True,
                rel_pos_zero_init=False,
                residual_pooling=True,
                dim_mul_in_att=True,
            )

            # if cfg.MODEL.ACT_CHECKPOINT:
            #     attention_block = checkpoint_wrapper(attention_block)
            self.blocks.append(attention_block)

            if len(stride_q[i]) > 0:
                input_size = [
                    size // stride for size, stride in zip(input_size, stride_q[i])
                ]

            embed_dim = dim_out
        #print("embed dim:"+str(embed_dim))
        self.norm = norm_layer(embed_dim)

        self.head = TransformerBasicHead(
            embed_dim,
            num_classes,
            dropout_rate=0.0,
            act_func="softmax",
        )
        if self.use_abs_pos:
            trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_embed_on:
            trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0.0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        names = []
        if self.zero_decay_pos_cls:
            # add all potential params
            names = ["pos_embed", "rel_pos_h", "rel_pos_w", "cls_token"]

        return names

    def forward(self, x):
        #print("inside MViT")
        #print("x shape before patch embed:"+str(x.shape))
        x, bchw = self.patch_embed(x)
        #print("x shape after patch embed:" + str(x.shape))
        H, W = bchw[-2], bchw[-1]

        #B, N, C = x.shape
        B, h, w, C = x.shape
        x = x.reshape(B, h*w, C)

        if self.cls_embed_on:
            cls_tokens = self.cls_token.expand(B, -1, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            #print("x shape after cls_tokens:" + str(x.shape))

        if self.use_abs_pos:
            x = x + self.pos_embed
            #print("x shape after use_abs_pos:" + str(x.shape))

        thw = [H, W]
        i = 0
        for blk in self.blocks:

            x, thw = blk(x, thw)
            #print("x shape after block " + str(i) + ":" + str(x.shape))
            i+=1
        H, W = thw
        # if x.ndim == 4:
        #     pass
        # elif x.ndim == 3:
        #     x = x.unsqueeze(1)
        B, N, C = x.shape
        #print("new x shape:"+str(x.shape))
        x = x.reshape(B, H, W, C)#.permute(0, 3, 1, 2).contiguous()
        x = self.norm(x)
        #print("x shape after norm:" + str(x.shape))
        # if self.cls_embed_on:
        #     x = x[:, 0]
        #     #print("x shape after cls_embed_on:" + str(x.shape))
        # else:
        #     x = x.mean(1)
        #     #print("x shape after mean:" + str(x.shape))

        x = self.head(x)
        #print("x shape after head:" + str(x.shape))
        return x


def _prepare_mvit_configs():#cfg):
    """
    Prepare mvit configs for dim_mul and head_mul facotrs, and q and kv pooling
    kernels and strides.
    """
    depth = 10
    dim_mul, head_mul = torch.ones(depth + 1), torch.ones(depth + 1)
    mvit_dim_mul = [[1, 2.0], [3, 2.0], [8, 2.0]]
    mvit_head_mul = [[1, 2.0], [3, 2.0], [8, 2.0]]
    for i in range(len(mvit_dim_mul)):
        dim_mul[mvit_dim_mul[i][0]] = mvit_dim_mul[i][1]
    for i in range(len(mvit_head_mul)):
        head_mul[mvit_head_mul[i][0]] = mvit_head_mul[i][1]

    pool_q = [[] for i in range(depth)]
    pool_kv = [[] for i in range(depth)]
    stride_q = [[] for i in range(depth)]
    stride_kv = [[] for i in range(depth)]
    mvit_pool_q_stride = [[0, 1, 1], [1, 2, 2], [2, 1, 1], [3, 2, 2], [4, 1, 1], [5, 1, 1], [6, 1, 1], [7, 1, 1],
                          [8, 2, 2], [9, 1, 1]]
    for i in range(len(mvit_pool_q_stride)):

        stride_q[mvit_pool_q_stride[i][0]] = mvit_pool_q_stride[i][1:]
        pool_q[mvit_pool_q_stride[i][0]] = [3,3]

    # If POOL_KV_STRIDE_ADAPTIVE is not None, initialize POOL_KV_STRIDE.
    mvit_pool_kv_stride_adaptive = [4,4]
    if mvit_pool_kv_stride_adaptive is not None:
        _stride_kv = mvit_pool_kv_stride_adaptive
        mvit_POOL_KV_STRIDE = []
        for i in range(10):
            if len(stride_q[i]) > 0:
                _stride_kv = [
                    max(_stride_kv[d] // stride_q[i][d], 1)
                    for d in range(len(_stride_kv))
                ]
            mvit_POOL_KV_STRIDE.append([i] + _stride_kv)

    for i in range(len(mvit_POOL_KV_STRIDE)):
        stride_kv[mvit_POOL_KV_STRIDE[i][0]] = mvit_POOL_KV_STRIDE[i][1:]
        pool_kv[mvit_POOL_KV_STRIDE[i][0]] = [3,3]

    return dim_mul, head_mul, pool_q, pool_kv, stride_q, stride_kv
