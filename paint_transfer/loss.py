from abc import ABCMeta, abstractmethod
import chainer
import numpy

from paint_transfer.config import LossConfig
from paint_transfer.forwarder import Forwarder
from paint_transfer.model import Models
from paint_transfer import utility


class BaseLoss(object, metaclass=ABCMeta):
    def __init__(self, config: LossConfig):
        self.config = config

    @staticmethod
    def blend_loss(loss, blend_config):
        assert sorted(loss.keys()) == sorted(blend_config.keys()), '{} {}'.format(loss.keys(), blend_config.keys())

        sum_loss = None

        for key in sorted(loss.keys()):
            blend = blend_config[key]
            if blend == 0.0:
                continue

            l = loss[key] * blend_config[key]

            if sum_loss is None:
                sum_loss = l
            else:
                sum_loss += l

        return sum_loss

    @abstractmethod
    def make_loss(self, *args, test, **kwargs):
        pass

    @abstractmethod
    def sum_loss(self, loss):
        pass

    @abstractmethod
    def test(self, *args, **kwargs):
        pass


class LossMaker(BaseLoss):
    def __init__(
            self,
            config: LossConfig,
            forwarder: Forwarder,
            models: Models,
    ):
        super().__init__(config)
        self.forwarder = forwarder
        self.models = models

    def make_loss(self, target, raw_line, test):
        xp = self.models.mismatch_discriminator.xp
        batchsize = target.shape[0]
        l_true = xp.ones(batchsize, dtype=numpy.float32)
        l_false = xp.zeros(batchsize, dtype=numpy.float32)

        raw_line_mismatch = chainer.functions.permutate(
            raw_line, indices=numpy.roll(numpy.arange(batchsize, dtype=numpy.int32), shift=1), axis=0)

        output = self.forwarder.forward(
            input=target,
            raw_line=raw_line,
            raw_line_mismatch=raw_line_mismatch,
            test=test,
        )
        generated = output['generated']
        match = output['match']
        mismatch = output['mismatch']
        z = output['z']

        mse = chainer.functions.mean_squared_error(generated, target)
        loss_gen = {'mse': mse}
        chainer.report(loss_gen, self.models.generator)

        match_lsm = utility.chainer.least_square_mean(match, l_false)
        mismatch_lsm = utility.chainer.least_square_mean(mismatch, l_true)
        loss_mismatch_discriminator = {'match_lsm': match_lsm, 'mismatch_lsm': mismatch_lsm}
        chainer.report(loss_mismatch_discriminator, self.models.mismatch_discriminator)

        fake_mismatch_lsm = utility.chainer.least_square_mean(match, l_true)
        z_l2 = chainer.functions.sum(z ** 2) / z.size
        loss_enc = {'mse': mse, 'fake_mismatch_lsm': fake_mismatch_lsm, 'activity_regularization': z_l2}
        chainer.report(loss_enc, self.models.encoder)

        return {
            'encoder': loss_enc,
            'generator': loss_gen,
            'mismatch_discriminator': loss_mismatch_discriminator,
        }

    def get_loss_names(self):
        return ['sum_loss'] + \
               list(self.config.blend['encoder'].keys()) + \
               list(self.config.blend['generator'].keys()) + \
               list(self.config.blend['mismatch_discriminator'].keys())

    def sum_loss(self, loss):
        sum_loss_enc = BaseLoss.blend_loss(loss['encoder'], self.config.blend['encoder'])
        chainer.report({'sum_loss': sum_loss_enc}, self.models.encoder)

        sum_loss_gen = BaseLoss.blend_loss(loss['generator'], self.config.blend['generator'])
        chainer.report({'sum_loss': sum_loss_gen}, self.models.generator)

        sum_loss_mismatch_discriminator = BaseLoss.blend_loss(
            loss['mismatch_discriminator'], self.config.blend['mismatch_discriminator'])
        chainer.report({'sum_loss': sum_loss_mismatch_discriminator}, self.models.mismatch_discriminator)

        return {
            'encoder': sum_loss_enc,
            'generator': sum_loss_gen,
            'mismatch_discriminator': sum_loss_mismatch_discriminator,
        }

    def test(self, target, raw_line):
        loss = self.make_loss(target, raw_line, test=True)
        return sum(self.sum_loss(loss).values())
