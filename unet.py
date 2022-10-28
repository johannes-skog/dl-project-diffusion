from mimetypes import init
import re
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
        channels: List[int],
        in_channels: int = 3,
        resnet_block_groups: int = 8,
        use_convnext: bool = True,
        convnext_mult: int = 2,
        init_channel_mult: int = 32,
    ):
        super().__init__()

        _channels = channels[:]

        # determine dimensions
        self._nchannels = in_channels

        _current_channels = in_channels

        self._init_conv = torch.nn.Conv2d(
            in_channels=_current_channels,
            out_channels=_current_channels * init_channel_mult,
            kernel_size=7,
            padding=3
        )

        _current_channels = _current_channels * init_channel_mult

        if use_convnext:
            _ConvBlock = partial(ConvNextBlock, mult=convnext_mult)
        else:
            _ConvBlock = partial(ResnetBlock, norm_groups=resnet_block_groups)

        time_dim = _current_channels * 4

        self._time_mlp = torch.nn.Sequential(
            SinusoidalPositionEmbeddings(_current_channels),
            torch.nn.Linear(_current_channels, time_dim),
            torch.nn.GELU(),
            torch.nn.Linear(time_dim, time_dim),
        )

        # layers
        self._down_sampling_layers = torch.nn.ModuleList([])
        self._up_sampling_layers = torch.nn.ModuleList([])

        n_layers = len(channels)

        for i, channels_out in enumerate(_channels):

            last_layer = i >= (n_layers - 1)

            self._down_sampling_layers.append(
                torch.nn.ModuleList(
                    [
                        _ConvBlock(_current_channels, channels_out, time_emb_dim=time_dim),
                        _ConvBlock(channels_out, channels_out, time_emb_dim=time_dim),
                        Residual(GroupPreNormalizer(channels_out, ConvLinearAttention(channels_out))),
                        # Downsample
                        (
                            torch.nn.Conv2d(
                                in_channels=channels_out,
                                out_channels=channels_out,
                                kernel_size=4,
                                stride=2,
                                padding=1
                            )
                            if not last_layer else torch.nn.Identity()
                        )
                    ]
                )
            )

            _current_channels = channels_out

        self.mid_block1 = _ConvBlock(_current_channels, _current_channels, time_emb_dim=time_dim)
        self.mid_attn = Residual(GroupPreNormalizer(_current_channels, ConvAttention(_current_channels)))
        self.mid_block2 = _ConvBlock(_current_channels, _current_channels, time_emb_dim=time_dim)

        _channels = [x for x in (reversed(_channels))][1:]

        for i, channels_out in enumerate(_channels):

            last_layer = i >= (n_layers - 1)

            self._up_sampling_layers.append(
                torch.nn.ModuleList(
                    [
                        _ConvBlock(_current_channels * 2, channels_out, time_emb_dim=time_dim),
                        _ConvBlock(channels_out, channels_out, time_emb_dim=time_dim),
                        Residual(GroupPreNormalizer(channels_out, ConvLinearAttention(channels_out))),
                        (
                            torch.nn.ConvTranspose2d(
                                in_channels=channels_out,
                                out_channels=channels_out,
                                kernel_size=4,
                                stride=2,
                                padding=1
                            )
                            if not last_layer else torch.nn.Identity()
                        )
                    ]
                )
            )

            _current_channels = channels_out

        self.final_conv = torch.nn.Sequential(
            _ConvBlock(_current_channels, _current_channels),
            torch.nn.Conv2d(_current_channels, in_channels, 1)
        )

    def forward(self, x, time):

        x = self._init_conv(x)

        t = self._time_mlp(time) if self._time_mlp is not None else None

        h = []

        # downsample
        for block1, block2, attn, downsample in self._down_sampling_layers:
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
        for block1, block2, attn, upsample in self._up_sampling_layers:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)

        return self.final_conv(x)


from utils import ClassEmbeddings

class UnetConditional(Unet):

    def __init__(
        self,
        channels: List[int],
        nclasses: int,
        in_channels: int = 3,
        resnet_block_groups: int = 8,
        use_convnext: bool = True,
        convnext_mult: int = 2,
        init_channel_mult: int = 32,
    ):

        super().__init__(
            channels=channels,
            in_channels=in_channels,
            resnet_block_groups=resnet_block_groups,
            use_convnext=use_convnext,
            convnext_mult=convnext_mult,
            init_channel_mult=init_channel_mult
        )

        self._nclasses = nclasses

        self._classes_embedder = ClassEmbeddings(
            nclasses=nclasses,
            out_dim=in_channels * init_channel_mult * 4
        )

    def forward(self, x: torch.Tensor, time: torch.Tensor, classes: torch.Tensor):

        x = self._init_conv(x)

        t = self._time_mlp(time) if self._time_mlp is not None else None
        c = self._classes_embedder(classes)

        e = t + c

        h = []

        # downsample
        for block1, block2, attn, downsample in self._down_sampling_layers:
            x = block1(x, e)
            x = block2(x, e)
            x = attn(x)
            h.append(x)
            x = downsample(x)


        # bottleneck
        x = self.mid_block1(x, e)
        x = self.mid_attn(x)
        x = self.mid_block2(x, e)

        # upsample
        for block1, block2, attn, upsample in self._up_sampling_layers:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, e)
            x = block2(x, e)
            x = attn(x)
            x = upsample(x)

        return self.final_conv(x)

