from quantizers.FiniteScalarQuantizer import FiniteScalarQuantizer
import jax
import equinox as eqx
import equinox.nn as nn
import jax.numpy as jnp
from utils.upsample import Upsample1d

class ResUnit(eqx.Module):
    conv1: jax.Array
    conv2: jax.Array

    def __init__(
        self, channel_in, channel_out, kernel_size, dilation, bias=True, key=None
    ):
        key1, key2 = jax.random.split(key)

        self.conv1 = nn.WeightNorm(
            nn.Conv1d(
                channel_in,
                channel_out,
                kernel_size,
                stride=1,
                dilation=dilation,
                padding="SAME",
                use_bias=bias,
                key=key1,
            )
        )
        self.conv2 = nn.WeightNorm(
            nn.Conv1d(
                channel_out,
                channel_out,
                kernel_size=1,
                stride=1,
                padding="SAME",
                use_bias=bias,
                key=key2,
            )
        )

    @eqx.filter_jit
    def __call__(self, x):
        y = self.conv1(jax.nn.elu(x))
        y = self.conv2(jax.nn.elu(y))
        return y + x


class ResBlock(eqx.Module):
    suite: nn.Sequential

    def __init__(
        self,
        channel_in,
        channel_out,
        kernel_size: int,
        stride: int,
        mode: str,
        dilations=(1, 3, 9),
        bias=True,
        key=None,
    ):
        key1, key2 = jax.random.split(key)

        res_channels = channel_in if mode == "encoder" else channel_out

        res_units = [
            ResUnit(
                res_channels,
                res_channels,
                kernel_size=kernel_size,
                bias=False,
                dilation=dilation,
                key=k,
            )
            for dilation, k in zip(dilations, jax.random.split(key1, len(dilations)))
        ]

        if mode == "encoder":
            if channel_in == channel_out:
                self.suite = nn.Sequential(
                    res_units
                    + [
                        nn.AvgPool1d(kernel_size=stride, stride=stride),
                    ]
                )
            else:
                self.suite = nn.Sequential(
                    res_units
                    + [
                        nn.WeightNorm(
                            nn.Conv1d(
                                channel_in,
                                channel_out,
                                kernel_size=(2 * stride),
                                stride=stride,
                                use_bias=bias,
                                padding="SAME",
                                key=key2,
                            )
                        )
                    ]
                )
        elif mode == "decoder":
            if channel_in == channel_out:
                self.suite = nn.Sequential(
                    [
                        Upsample1d(scale_factor=stride, mode="nearest"),
                    ]
                    + res_units
                )
            else:
                self.suite = nn.Sequential(
                    [
                        nn.WeightNorm(
                            nn.ConvTranspose1d(
                                channel_in,
                                channel_out,
                                kernel_size=(2 * stride),
                                stride=stride,
                                use_bias=bias,
                                padding="SAME",
                                key=key2,
                            )
                        )
                    ]
                    + res_units
                )

    @eqx.filter_jit
    def __call__(self, x):
        out = x
        for unit in self.suite:
            out = unit(out)
        return out


class Encoder(eqx.Module):
    suite: list

    def __init__(
        self,
        in_channels,
        hidden_channels,
        latent_channels,
        kernel_size=5,
        channel_ratios: tuple = (1, 4, 8, 8, 16, 16),
        strides: tuple = (2, 2, 2, 5, 5),
        key=None,
    ):
        key0, key1, key2 = jax.random.split(key, 3)
        self.suite = (
            [
                nn.WeightNorm(
                    nn.Conv1d(
                        in_channels,
                        hidden_channels * channel_ratios[0],
                        kernel_size=kernel_size,
                        stride=1,
                        padding="SAME",
                        use_bias=False,
                        key=key0,
                    )
                )
            ]
            + [
                ResBlock(
                    hidden_channels * channel_ratios[idx],
                    hidden_channels * channel_ratios[idx + 1],
                    kernel_size=kernel_size,
                    stride=strides[idx],
                    mode="encoder",
                    bias=True,
                    key=k,
                )
                for idx, k in enumerate(jax.random.split(key1, len(strides)))
            ]
            + [
                nn.WeightNorm(
                    nn.Conv1d(
                        hidden_channels * channel_ratios[-1],
                        latent_channels,
                        kernel_size=1,
                        padding="SAME",
                        key=key2,
                    )
                )
            ]
        )

    @eqx.filter_jit
    def __call__(self, x):
        out = x

        for unit in self.suite:
            out = unit(out)
            # jax.debug.print("{x}", x=out[0][:10])
            out = jax.nn.elu(out)

        return out


class Decoder(eqx.Module):
    suite: list

    def __init__(
        self,
        in_channels,
        hidden_channels,
        latent_channels,
        kernel_size=5,
        channel_ratios: tuple = (16, 16, 8, 8, 4, 1),
        strides: tuple = (5, 5, 2, 2, 2),
        key=None,
    ):
        key0, key1, key2 = jax.random.split(key, 3)
        self.suite = (
            [
                nn.WeightNorm(
                    nn.ConvTranspose1d(
                        latent_channels,
                        hidden_channels * channel_ratios[0],
                        kernel_size=1,
                        padding="SAME",
                        use_bias=True,
                        key=key0,
                    )
                )
            ]
            + [
                ResBlock(
                    hidden_channels * channel_ratios[idx],
                    hidden_channels * channel_ratios[idx + 1],
                    kernel_size=kernel_size,
                    stride=strides[idx],
                    mode="decoder",
                    bias=True,
                    key=k,
                )
                for idx, k in enumerate(jax.random.split(key1, len(strides)))
            ]
            + [
                nn.WeightNorm(
                    nn.ConvTranspose1d(
                        hidden_channels * channel_ratios[-1],
                        in_channels,
                        kernel_size=kernel_size,
                        stride=1,
                        padding="SAME",
                        key=key2,
                    )
                )
            ]
        )

    @eqx.filter_jit
    def __call__(self, x):
        out = x
        for unit in self.suite:
            out = unit(out)
            # jax.debug.print("{x}", x=out[0][:10])
            out = jax.nn.elu(out)

        return out

class FSQVAE(eqx.Module):
    encoder: Encoder
    decoder: Decoder
    quantizer: FiniteScalarQuantizer

    def __init__(self, in_channels, hidden_channels, latent_channels, levels, key=None):
        key1, key2 = jax.random.split(key)

        self.encoder = Encoder(in_channels, hidden_channels, latent_channels, key=key1)
        self.decoder = Decoder(in_channels, hidden_channels, latent_channels, key=key2)
        self.quantizer = FiniteScalarQuantizer(levels=levels)

    @eqx.filter_jit
    def __call__(self, x):

        z_e = self.encoder(x)
        reshaped_z_e = jnp.reshape(
            z_e, (-1, 5)
        )  # 16000 Hz -> 16Hz = 1000 points per code downsampled from 1000 to 5. Map each set of 5 to their respective code and map back. There are 4 levels thus 5 * (2 ** 4) = 80bits per codeword
        reshaped_z_q = self.quantizer(reshaped_z_e)
        z_q = jnp.reshape(reshaped_z_q, z_e.shape)
        y = self.decoder(z_q)
        return y
