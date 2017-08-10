import chainer
from collections import namedtuple
import typing

from paint_transfer.config import ModelConfig
from paint_transfer.network import *
from paint_transfer import utility

Models = namedtuple('Models', ['encoder', 'generator', 'mismatch_discriminator'])


class BaseModel(chainer.Chain, metaclass=ABCMeta):
    def __init__(self, config: ModelConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config

    @abstractmethod
    def __call__(self, x, test) -> (chainer.Variable, typing.Dict):
        pass


class DeepEncoder(BaseModel):
    def __init__(self, config: ModelConfig):
        num_z = config.base_num_z_encoder
        super().__init__(
            config,
            conv=DeepConvolution(num_scale=5, base_num_z=num_z),
            post=utility.chainer.Link.create_convolution_2d(None, 512, ksize=4, stride=4),
        )

    def __call__(self, x, test):
        h = x
        h = getattr(self, 'conv')(h, test)
        h = getattr(self, 'post')(h)
        assert h.shape[2] == h.shape[3] == 1
        return h, {}


class ResidualGenerator(BaseModel):
    def __init__(self, config: ModelConfig):
        num_z = config.base_num_z_generator
        super().__init__(
            config,
            unet_encoder=UnetEncoder(base_num_z=num_z),
            unet_decoder=UnetDecoder(base_num_z=num_z),
            post=utility.chainer.Link.create_convolution_2d(None, 3, ksize=3, stride=1, pad=1),
        )

    def __call__(self, x, test):
        z, raw_line = x['z'], x['raw_line']

        h = raw_line
        h = getattr(self, 'unet_encoder')(h, test)

        z = chainer.functions.unpooling_2d(z, ksize=h[-1].shape[-2:], cover_all=False)
        h[-1] = chainer.functions.concat([h[-1], z], axis=1)

        h = getattr(self, 'unet_decoder')(h, test)
        h = getattr(self, 'post')(h)
        return h, {}


class DeepMismatchDiscriminator(BaseModel):
    def __init__(self, config: ModelConfig):
        num_z = config.base_num_z_discriminator
        super().__init__(
            config,
            conv=DeepConvolution(num_scale=5, base_num_z=num_z),
            post_conv=utility.chainer.Link.create_convolution_2d(None, num_z * 2 ** (5 - 1), ksize=1, nobias=True),
            post_bn=chainer.links.BatchNormalization(num_z * 2 ** (5 - 1)),
            post_linear=utility.chainer.Link.create_linear(None, 1),
        )

    def __call__(self, x, test):
        z, raw_line = x['z'], x['raw_line']

        h = raw_line
        h = getattr(self, 'conv')(h, test)

        z = chainer.functions.unpooling_2d(z, ksize=h.shape[-2:], cover_all=False)
        h = chainer.functions.concat([h, z], axis=1)

        h = chainer.functions.leaky_relu(getattr(self, 'post_bn')(getattr(self, 'post_conv')(h), test))
        h = chainer.functions.squeeze(getattr(self, 'post_linear')(h))
        return h, {}


def create_model(config: ModelConfig) -> Models:
    return Models(
        encoder=DeepEncoder(config),
        generator=ResidualGenerator(config),
        mismatch_discriminator=DeepMismatchDiscriminator(config),
    )
