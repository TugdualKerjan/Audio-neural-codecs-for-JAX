
from quantizers.VectorQuantizer import VectorQuantizer
import jax
import equinox as eqx
import equinox.nn as nn
import typing as tp




class ResBlock(eqx.Module):
    conv1: nn.Conv1d
    conv2: nn.Conv1d
    conv3: nn.Conv1d
    act: tp.Callable = eqx.static_field()

    def __init__(self, dim: int, activation=jax.nn.relu, key=None):

        key1, key2, key3 = jax.random.split(key, 3)

        self.conv1 = nn.Conv1d(dim, dim, kernel_size=3, padding=(1,), key=key1)
        self.conv2 = nn.Conv1d(dim, dim, kernel_size=3, padding=(1,), key=key2)
        self.conv3 = nn.Conv1d(dim, dim, kernel_size=1, key=key3)

        self.act = activation

    def __call__(self, x):
        y = x

        y = self.conv1(y)
        y = jax.nn.relu(y)
        y = self.conv2(y)
        y = jax.nn.relu(y)
        y = self.conv3(y)

        y = y + x

        return y




class Encoder(eqx.Module):
    conv1: nn.Conv1d
    conv2: nn.Conv1d
    conv3: nn.Conv1d
    res1: ResBlock
    res2: ResBlock
    res3: ResBlock

    def __init__(self, hidden_dim: int = 1024, codebook_dim: int = 512, key=None):
        key1, key2, key3, key4, key5, key6 = jax.random.split(key, 6)

        self.conv1 = nn.Conv1d(
            in_channels=80,
            out_channels=512,
            kernel_size=3,
            stride=2,
            padding=(1,),
            key=key1,
        )
        self.conv2 = nn.Conv1d(
            in_channels=512,
            out_channels=hidden_dim,
            kernel_size=3,
            stride=2,
            padding=(1,),
            key=key2,
        )
        self.res1 = ResBlock(dim=hidden_dim, key=key3)
        self.res2 = ResBlock(dim=hidden_dim, key=key4)
        self.res3 = ResBlock(dim=hidden_dim, key=key5)
        self.conv3 = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=codebook_dim,
            kernel_size=1,
            stride=1,
            # padding="SAME",
            key=key6,
        )

    def __call__(self, x):

        y = self.conv1(x)
        y = jax.nn.relu(y)
        y = self.conv2(y)
        y = jax.nn.relu(y)
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.conv3(y)

        return y




# | label: upsample


class UpsampledConv(eqx.Module):
    conv: nn.Conv1d
    stride: int = eqx.static_field()

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tp.Union[int, tp.Tuple[int]],
        stride: int,
        padding: tp.Union[int, str],
        key=None,
    ):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            key=key,
        )

    def __call__(self, x):
        upsampled_size = (x.shape[0], x.shape[1] * self.stride)
        upsampled = jax.image.resize(x, upsampled_size, method="nearest")
        return self.conv(upsampled)




class Decoder(eqx.Module):
    conv1: nn.Conv1d
    conv2: UpsampledConv
    conv3: UpsampledConv
    conv4: nn.Conv1d
    res1: ResBlock
    res2: ResBlock
    res3: ResBlock

    def __init__(self, hidden_dim: int = 1024, codebook_dim: int = 512, key=None):
        key1, key2, key3, key4, key5, key6, key7 = jax.random.split(key, 7)

        self.conv1 = nn.Conv1d(
            in_channels=codebook_dim,
            out_channels=hidden_dim,
            kernel_size=1,
            stride=1,
            # padding="SAME",
            key=key1,
        )
        self.res1 = ResBlock(dim=hidden_dim, key=key2)
        self.res2 = ResBlock(dim=hidden_dim, key=key3)
        self.res3 = ResBlock(dim=hidden_dim, key=key4)
        self.conv2 = UpsampledConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=3,
            stride=2,
            padding=(1,),
            key=key5,
        )
        self.conv3 = UpsampledConv(
            in_channels=hidden_dim,
            out_channels=512,
            kernel_size=3,
            stride=2,
            padding=(1,),
            key=key6,
        )
        self.conv4 = nn.Conv1d(
            in_channels=512,
            out_channels=80,
            kernel_size=1,
            stride=1,
            # padding="SAME",
            key=key7,
        )

    def __call__(self, x):

        y = self.conv1(x)
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.conv2(y)
        y = jax.nn.relu(y)
        y = self.conv3(y)
        y = jax.nn.relu(y)
        y = self.conv4(y)

        return y
    
    
    class VQVAE(eqx.Module):
        encoder: Encoder
        decoder: Decoder
        quantizer: VectorQuantizer

        def __init__(self, key=None):
            key1, key2, key3 = jax.random.split(key, 3)

            self.encoder = Encoder(key=key1)
            self.decoder = Decoder(key=key2)
            self.quantizer = VectorQuantizer(decay=0.8, key=key3)

        def __call__(self, x):
            z_e = self.encoder(x)
            z_q, codebook_indices = self.quantizer(z_e)
            y = self.decoder(z_q)

            return z_e, z_q, codebook_indices, y