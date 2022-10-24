import torch
import einops

class ConvBlock(torch.nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        norm_groups: int  = 8,
    ):

        super().__init__()

        self._conv = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding = 1
        )

        self._gnorm = torch.nn.GroupNorm(
            num_groups=norm_groups,
            num_channels=out_channels,
        )

        self._actfunc = torch.nn.SiLU()

    def forward(self, x: torch.Tensor):

        x = self.proj(x)
        x = self.norm(x)
        x = self.act(x)

        return x

class ResnetBlock(torch.nn.Module):
    """https://arxiv.org/abs/1512.03385"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int = None,
        norm_groups: int = 8,
        kernel_size: int = 3,
    ):
        super().__init__()

        self._time_embedder = (
            torch.nn.Sequential(
                torch.nn.SiLU(),
                torch.nn.Linear(time_emb_dim, out_channels),
            )
            if (time_emb_dim is not None) else None
        )

        self._block1 = ConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            norm_groups=norm_groups,
            kernel_size=kernel_size,
        )

        self._block2 = ConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            norm_groups=norm_groups,
            kernel_size=kernel_size
        )

        self._res_conv = (
            torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
            if in_channels != out_channels else torch.nn.Identity()
        )

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor = None):

        y = self._block1(x)

        # Add the time ebedding info
        if (self._time_embedder is not None) and (time_emb is not None):
            time_emb = self._time_embedder(time_emb)
            h = einops.rearrange(time_emb, "b c -> b c 1 1") + y

        y = self._block2(y)

        return y + self.res_conv(x)


class ConvNextBlock(torch.nn.Module):
    """https://arxiv.org/abs/2201.03545"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int = None,
        mult: int = 2,
        normalize: bool = True
    ):
        super().__init__()

        self._time_embedder = (
            torch.nn.Sequential(torch.nn.GELU(), torch.nn.Linear(time_emb_dim, in_channels))
            if (time_emb_dim is not None)
            else None
        )

        # Depthwise convolution, large kz w/o large impact on # of weights
        self._dw_conv = torch.nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=7,
            padding=3,
            groups=in_channels
        )

        self._convs = torch.nn.Sequential(
            torch.nn.GroupNorm(1, in_channels) if normalize else torch.nn.Identity(),
            torch.nn.Conv2d(in_channels, out_channels * mult, kernel_size=3, padding=1),
            torch.nn.GELU(),
            torch.nn.GroupNorm(num_groups=1, num_channels=out_channels * mult),
            torch.nn.Conv2d(out_channels * mult, out_channels, kernel_size=3, padding=1),
        )

        # If we have different channels for out and in, we will setup a single conv op to
        # bring the dims to be the same between input and output, the residual connection
        # is however not going to be direct in such a case
        self._res_conv = (
            torch.nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels else torch.nn.Identity()
        )

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor):

        y = self._dw_conv(x)

        if (self._time_embedder is not None) and (time_emb is not None):
            time_emb = self._time_embedder(time_emb)
            y = y + einops.rearrange(time_emb, "b c -> b c 1 1")

        y = self._convs(y)

        return y + self._res_conv(x)


class ConvAttention(torch.nn.Module):

    def __init__(self, in_channels: int, heads: int = 4, dim_head: int = 32):

        super().__init__()

        self._scale = dim_head**-0.5
        self._nheads = heads
        self._qkv_net = torch.nn.Conv2d(
            in_channels,
            dim_head * self._nheads * 3, # for the q k v
            kernel_size=1,
            bias=False
        )
        self._out_net = torch.nn.Conv2d(dim_head * heads, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor):

        b, c, h, w = x.shape

        # split up the channels into three, for q, k, v
        qkv = self._qkv_net(x).chunk(3, dim=1)

        # Split up the heads to a seperate dim, flatten height and weigth into a single dim
        q, k, v = map(
            lambda t: einops.rearrange(t, "b (h c) x y -> b h c (x y)", h=self._nheads), qkv
        )

        q = q * self._scale

        # d is the channel dim, i, j are the pixle coords (flatten)
        sim = torch.einsum("b h d i, b h d j -> b h i j", q, k)
        # Scale to help out with num-stability
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = torch.einsum("b h i j, b h d j -> b h i d", attn, v)
        out = einops.rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)

        return self._out_net(out)

class ConvLinearAttention(torch.nn.Module):

    def __init__(self, in_channels: int, heads: int = 4, dim_head: int = 32):

        super().__init__()

        self._scale = dim_head**-0.5

        self._heads = heads

        self._qkv_net = torch.nn.Conv2d(in_channels, dim_head * heads * 3, 1, bias=False)

        self._out_net = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=dim_head * heads, out_channels=in_channels, kernel_size=1),
            torch.nn.GroupNorm(num_groups=1, num_channels=in_channels)
        )

    def forward(self, x: torch.Tensor):

        b, c, h, w = x.shape
        qkv = self.to__qkv_netqkv(x).chunk(3, dim=1)

        q, k, v = map(
            lambda t: einops.rearrange(t, "b (h c) x y -> b h c (x y)", h=self._heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self._scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = einops.rearrange(out, "b h c (x y) -> b (h c) x y", h=self._heads, x=h, y=w)

        return self._out_net(out)