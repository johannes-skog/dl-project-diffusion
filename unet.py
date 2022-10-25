from typing import List
import torch
from functools import partial
from utils import (
    ConvBlock,
    ResnetBlock,
    ConvNextBlock,
    ConvAttention,
    ConvLinearAttention,
    SinusoidalPositionEmbeddings,
    Residual,
    GroupPreNormalizer,
)


class Unet(torch.nn.Module):
    def __init__(
        self,
        scale_channels: int,
        init_channels: int,
        out_channels: int,
        nchannels: int = 3,
        dim_mults: List[int] = (1, 2, 4, 8),
        resnet_block_groups: int = 8,
        use_convnext: bool = True,
        convnext_mult: int = 2,
    ):
        super().__init__()

        # determine dimensions
        self._nchannels = nchannels

        self._init_conv = torch.nn.Conv2d(
            in_channels=nchannels,
            out_channels=init_channels,
            kernel_size=7,
            padding=3
        )

        dims = [init_channels, *map(lambda m: scale_channels * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        if use_convnext:
            _ConvBlock = partial(ConvNextBlock, mult=convnext_mult)
        else:
            _ConvBlock = partial(ResnetBlock, groups=resnet_block_groups)

        time_dim = scale_channels * 4
        self.time_mlp = torch.nn.Sequential(
            SinusoidalPositionEmbeddings(scale_channels),
            torch.nn.Linear(scale_channels, time_dim),
            torch.nn.GELU(),
            torch.nn.Linear(time_dim, time_dim),
        )

        # layers
        self.downs = torch.nn.ModuleList([])
        self.ups = torch.nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                torch.nn.ModuleList(
                    [
                        _ConvBlock(dim_in, dim_out, time_emb_dim=time_dim),
                        _ConvBlock(dim_out, dim_out, time_emb_dim=time_dim),
                        Residual(GroupPreNormalizer(dim_out, ConvLinearAttention(dim_out))),
                        # Downsample
                        (
                            torch.nn.Conv2d(
                                in_channels=dim_out,
                                out_channels=dim_out,
                                kernel_size=4,
                                stride=2,
                                padding=1
                            )
                            if not is_last else torch.nn.Identity()
                        )
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = _ConvBlock(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(GroupPreNormalizer(mid_dim, ConvAttention(mid_dim)))
        self.mid_block2 = _ConvBlock(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):

            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                torch.nn.ModuleList(
                    [
                        _ConvBlock(dim_out * 2, dim_in, time_emb_dim=time_dim),
                        _ConvBlock(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(GroupPreNormalizer(dim_in, ConvLinearAttention(dim_in))),
                        (
                            torch.nn.ConvTranspose2d(
                                in_channels=dim_in,
                                out_channels=dim_in,
                                kernel_size=4,
                                stride=2,
                                padding=1
                            )
                            if not is_last else torch.nn.Identity()
                        )
                    ]
                )
            )

        out_dim = out_channels if out_channels is not None else nchannels
        self.final_conv = torch.nn.Sequential(
            _ConvBlock(scale_channels, scale_channels),
            torch.nn.Conv2d(scale_channels, out_dim, 1)
        )

    def forward(self, x, time):

        x = self._init_conv(x)

        t = None # self.time_mlp(time) if self.time_mlp is not None else None

        h = []

        # downsample
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        # bottleneck
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        # upsample
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)

        return self.final_conv(x)