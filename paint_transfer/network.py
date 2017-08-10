from abc import ABCMeta, abstractmethod
import chainer

from paint_transfer import utility


class BaseNetwork(chainer.Chain, metaclass=ABCMeta):
    def __init__(self, base_activation=None, last_activation=None, **kwargs):
        super().__init__(**kwargs)

        self.base_activation = base_activation if base_activation is not None else chainer.functions.leaky_relu
        self.last_activation = last_activation if last_activation is not None else self.base_activation

    @abstractmethod
    def __call__(self, x, test: bool):
        pass

    def activation(self, h, is_last=False):
        if not is_last:
            return self.base_activation(h)
        else:
            return self.last_activation(h)


class BaseResidualBlock(BaseNetwork):
    def _padding_channel(self, h, x, test):
        if x.shape != h.shape:
            n, c, hh, ww = x.shape
            pad_c = h.shape[1] - c
            p = self.xp.zeros((n, pad_c, hh, ww), dtype=self.xp.float32)
            p = chainer.Variable(p, volatile='auto')
            x = chainer.functions.concat((p, x))
        return x


class ResidualBlock(BaseResidualBlock):
    def __init__(self, num_layer):
        super().__init__(
            conv1=utility.chainer.Link.create_convolution_2d(
                None, num_layer, ksize=3, stride=1, pad=1, nobias=True),
            bn1=chainer.links.BatchNormalization(num_layer),
            conv2=utility.chainer.Link.create_convolution_2d(
                num_layer, num_layer, ksize=3, stride=1, pad=1, nobias=True),
            bn2=chainer.links.BatchNormalization(num_layer),
        )

    def __call__(self, x, test):
        h = x
        h = self.activation(self.bn1(self.conv1(h), test=test))
        h = self.bn2(self.conv2(h), test=test)
        x = self._padding_channel(h, x, test)
        return x + h


class DilateResidualBlock(BaseResidualBlock):
    def __init__(self, num_layer, dilate):
        super().__init__(
            dilate=utility.chainer.Link.create_dilated_convolution_2d(
                None, num_layer, 3, 1, dilate, dilate=dilate, nobias=True),
            bn1=chainer.links.BatchNormalization(num_layer),
            conv=utility.chainer.Link.create_convolution_2d(
                num_layer, num_layer, ksize=3, stride=1, pad=1, nobias=True),
            bn2=chainer.links.BatchNormalization(num_layer),
        )

    def __call__(self, x, test):
        h = x
        chainer.functions.batch_l2_norm_squared()
        h = self.activation(self.bn1(self.dilate(h), test=test))
        h = self.bn2(self.conv(h), test=test)
        x = self._padding_channel(h, x, test)
        return x + h


class DeepResidual(BaseNetwork):
    def __init__(self, num_z, num_residual):
        super().__init__()
        self.num_residual = num_residual

        for i in range(num_residual):
            self.add_link('res{}'.format(i), ResidualBlock(num_z))

    def __call__(self, x, test):
        h = x
        for i in range(self.num_residual):
            h = getattr(self, 'res{}'.format(i))(h, test)
        return h


class DeepDilateResidual(BaseNetwork):
    def __init__(self, num_z, num_residual):
        super().__init__()
        self.num_residual = num_residual

        for i in range(num_residual):
            dilate = 2 ** (i + 1)
            self.add_link('res{}'.format(i), DilateResidualBlock(num_z, dilate))

    def __call__(self, x, test):
        h = x
        for i in range(self.num_residual):
            h = getattr(self, 'res{}'.format(i))(h, test)
        return h


class DeepConvolution(BaseNetwork):
    def __init__(self, num_scale, base_num_z, **kwargs):
        super().__init__(**kwargs)
        self.num_scale = num_scale

        for i in range(num_scale):
            l = base_num_z * 2 ** i
            self.add_link('conv{}'.format(i + 1),
                          utility.chainer.Link.create_convolution_2d(None, l, 4, 2, 1, nobias=True))
            self.add_link('bn{}'.format(i + 1), chainer.links.BatchNormalization(l))

    def get_scaled_width(self, base_width):
        return base_width // (2 ** self.num_scale)

    def __call__(self, x, test):
        h = x
        for i in range(self.num_scale):
            conv = getattr(self, 'conv{}'.format(i + 1))
            bn = getattr(self, 'bn{}'.format(i + 1))
            h = super().activation(bn(conv(h), test=test), is_last=(i == self.num_scale - 1))
        return h


class DeepDeconvolution(BaseNetwork):
    def __init__(self, num_scale, base_num_z, **kwargs):
        super().__init__(**kwargs)
        self.num_scale = num_scale

        for i in range(num_scale):
            l = base_num_z * 2 ** (num_scale - 1 - i)
            self.add_link('deconv{}'.format(i + 1),
                          utility.chainer.Link.create_deconvolution_2d(None, l, 4, 2, 1, nobias=True))
            self.add_link('bn{}'.format(i + 1), chainer.links.BatchNormalization(l))

    def get_scaled_width(self, base_width):
        return base_width // (2 ** self.num_scale)

    def __call__(self, x, test):
        h = x
        for i in range(self.num_scale):
            deconv = getattr(self, 'deconv{}'.format(i + 1))
            bn = getattr(self, 'bn{}'.format(i + 1))
            h = super().activation(bn(deconv(h), test=test), is_last=(i == self.num_scale - 1))
        return h


class UnetEncoder(BaseNetwork):
    def __init__(self, base_num_z):
        super().__init__(
            conv0=utility.chainer.Link.create_convolution_2d(None, base_num_z * 2 ** 0, 3, 1, 1, nobias=True),
            conv1=utility.chainer.Link.create_convolution_2d(None, base_num_z * 2 ** 1, 4, 2, 1, nobias=True),
            conv2=utility.chainer.Link.create_convolution_2d(None, base_num_z * 2 ** 1, 3, 1, 1, nobias=True),
            conv3=utility.chainer.Link.create_convolution_2d(None, base_num_z * 2 ** 2, 4, 2, 1, nobias=True),
            conv4=utility.chainer.Link.create_convolution_2d(None, base_num_z * 2 ** 2, 3, 1, 1, nobias=True),
            conv5=utility.chainer.Link.create_convolution_2d(None, base_num_z * 2 ** 3, 4, 2, 1, nobias=True),
            conv6=utility.chainer.Link.create_convolution_2d(None, base_num_z * 2 ** 3, 3, 1, 1, nobias=True),
            conv7=utility.chainer.Link.create_convolution_2d(None, base_num_z * 2 ** 4, 4, 2, 1, nobias=True),
            conv8=utility.chainer.Link.create_convolution_2d(None, base_num_z * 2 ** 4, 3, 1, 1, nobias=True),
            bn0=chainer.links.BatchNormalization(base_num_z * 2 ** 0),
            bn1=chainer.links.BatchNormalization(base_num_z * 2 ** 1),
            bn2=chainer.links.BatchNormalization(base_num_z * 2 ** 1),
            bn3=chainer.links.BatchNormalization(base_num_z * 2 ** 2),
            bn4=chainer.links.BatchNormalization(base_num_z * 2 ** 2),
            bn5=chainer.links.BatchNormalization(base_num_z * 2 ** 3),
            bn6=chainer.links.BatchNormalization(base_num_z * 2 ** 3),
            bn7=chainer.links.BatchNormalization(base_num_z * 2 ** 4),
            bn8=chainer.links.BatchNormalization(base_num_z * 2 ** 4),
        )

    def __call__(self, x, test):
        h = x
        h_list = []
        for i in range(9):
            conv = getattr(self, 'conv{}'.format(i))
            bn = getattr(self, 'bn{}'.format(i))
            h = super().activation(bn(conv(h), test=test))
            h_list.append(h)
        return h_list


class UnetDecoder(BaseNetwork):
    def __init__(self, base_num_z):
        super().__init__(
            deconv8=utility.chainer.Link.create_deconvolution_2d(None, base_num_z * 2 ** 4, 4, 2, 1, nobias=True),
            deconv7=utility.chainer.Link.create_deconvolution_2d(None, base_num_z * 2 ** 3, 3, 1, 1, nobias=True),
            deconv6=utility.chainer.Link.create_deconvolution_2d(None, base_num_z * 2 ** 3, 4, 2, 1, nobias=True),
            deconv5=utility.chainer.Link.create_deconvolution_2d(None, base_num_z * 2 ** 2, 3, 1, 1, nobias=True),
            deconv4=utility.chainer.Link.create_deconvolution_2d(None, base_num_z * 2 ** 2, 4, 2, 1, nobias=True),
            deconv3=utility.chainer.Link.create_deconvolution_2d(None, base_num_z * 2 ** 1, 3, 1, 1, nobias=True),
            deconv2=utility.chainer.Link.create_deconvolution_2d(None, base_num_z * 2 ** 1, 4, 2, 1, nobias=True),
            deconv1=utility.chainer.Link.create_deconvolution_2d(None, base_num_z * 2 ** 0, 3, 1, 1, nobias=True),
            bn8=chainer.links.BatchNormalization(base_num_z * 2 ** 4),
            bn7=chainer.links.BatchNormalization(base_num_z * 2 ** 3),
            bn6=chainer.links.BatchNormalization(base_num_z * 2 ** 3),
            bn5=chainer.links.BatchNormalization(base_num_z * 2 ** 2),
            bn4=chainer.links.BatchNormalization(base_num_z * 2 ** 2),
            bn3=chainer.links.BatchNormalization(base_num_z * 2 ** 1),
            bn2=chainer.links.BatchNormalization(base_num_z * 2 ** 1),
            bn1=chainer.links.BatchNormalization(base_num_z * 2 ** 0),
        )

    def __call__(self, x, test):
        e0, e1, e2, e3, e4, e5, e6, e7, e8 = x
        d8 = super().activation(self.bn8(self.deconv8(chainer.functions.concat([e7, e8])), test=test))
        d7 = super().activation(self.bn7(self.deconv7(d8), test=test))
        d6 = super().activation(self.bn6(self.deconv6(chainer.functions.concat([e6, d7])), test=test))
        d5 = super().activation(self.bn5(self.deconv5(d6), test=test))
        d4 = super().activation(self.bn4(self.deconv4(chainer.functions.concat([e4, d5])), test=test))
        d3 = super().activation(self.bn3(self.deconv3(d4), test=test))
        d2 = super().activation(self.bn2(self.deconv2(chainer.functions.concat([e2, d3])), test=test))
        d1 = super().activation(self.bn1(self.deconv1(d2), test=test))
        return chainer.functions.concat([e0, d1])
