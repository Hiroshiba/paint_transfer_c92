import argparse
import chainer

from paint_transfer.config import Config
from paint_transfer import dataset
from paint_transfer.forwarder import Forwarder
from paint_transfer.loss import LossMaker
from paint_transfer.model import create_model
from paint_transfer.updater import Updater
from paint_transfer.trainer import create_optimizer, create_trainer
from paint_transfer import utility

parser = argparse.ArgumentParser()
parser.add_argument('config_json_path')
config_json_path = parser.parse_args().config_json_path

# load config
config = Config(config_json_path)
config.copy_config_json()
train_config = config.train_config

nb = train_config.batchsize
use_gpu = (train_config.gpu >= 0)

# setup trainer
use_gpu and chainer.cuda.get_device(train_config.gpu).use()
utility.chainer.set_default_initialW(train_config.initial_weight['default'])

# setup dataset
datasets = dataset.choose(config.dataset_config)
IteratorClass = chainer.iterators.MultiprocessIterator
iterator_train = IteratorClass(datasets['train'], nb, repeat=True, shuffle=True)
iterator_test = IteratorClass(datasets['test'], nb, repeat=False, shuffle=False)
iterator_train_eval = IteratorClass(datasets['train_eval'], nb, repeat=False, shuffle=False)

# setup model
models = create_model(config.model_config)
if use_gpu:
    models.encoder.to_gpu()
    models.generator.to_gpu()
    models.mismatch_discriminator.to_gpu()
model_list = {
    'encoder': models.encoder,
    'generator': models.generator,
    'mismatch_discriminator': models.mismatch_discriminator,
}

optimizers = {}
for key, model in model_list.items():
    optimizer = create_optimizer(train_config, model, key)
    optimizers[key] = optimizer

# setup forwarder
forwarder = Forwarder(config.model_config, models)

# setup loss
loss_maker = LossMaker(config.loss_config, forwarder, models)

# setup updater
updater = Updater(
    optimizer=optimizers,
    iterator=iterator_train,
    loss_maker=loss_maker,
    device=train_config.gpu,
    converter=utility.chainer.converter_recursive,
)

# trainer
trainer = create_trainer(
    config=train_config,
    project_path=config.project_config.get_project_path(),
    updater=updater,
    model=model_list,
    eval_func=loss_maker.test,
    iterator_test=iterator_test,
    iterator_train_eval=iterator_train_eval,
    converter=utility.chainer.converter_recursive,
    loss_names=loss_maker.get_loss_names(),
)
trainer.run()
