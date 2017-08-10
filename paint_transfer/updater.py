import chainer

from paint_transfer.loss import LossMaker


class Updater(chainer.training.StandardUpdater):
    def __init__(self, loss_maker: LossMaker, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_maker = loss_maker

    def update_core(self):
        optimizers = self.get_all_optimizers()

        batch = self.converter(self.get_iterator('main').next(), self.device)
        loss = self.loss_maker.make_loss(**batch, test=False)

        sum_loss = self.loss_maker.sum_loss(loss)
        optimizers['encoder'].update(lambda sum_loss: sum_loss['encoder'], sum_loss)
        optimizers['generator'].update(lambda sum_loss: sum_loss['generator'], sum_loss)
        optimizers['mismatch_discriminator'].update(lambda sum_loss: sum_loss['mismatch_discriminator'], sum_loss)
