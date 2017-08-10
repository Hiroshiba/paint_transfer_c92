import chainer
from chainer.training import extensions
import typing

from paint_transfer.config import TrainConfig
from paint_transfer import utility


def create_optimizer(train_config: TrainConfig, model, model_name: str):
    config = train_config.optimizer[model_name]
    if config == 'default':
        config = train_config.optimizer['default']

    optimizer = None
    if config['name'] == 'adam':
        alpha = 0.001 if 'alpha' not in config else config['alpha']
        beta1 = 0.9 if 'beta1' not in config else config['beta1']
        beta2 = 0.999 if 'beta2' not in config else config['beta2']
        optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1, beta2=beta2)
    elif config['name'] == 'RMSprop':
        lr = 0.01 if 'lr' not in config else config['lr']
        alpha = 0.99 if 'alpha' not in config else config['alpha']
        optimizer = chainer.optimizers.RMSprop(lr=lr, alpha=alpha)
    else:
        assert "{name} is not defined.".format(name=config['name'])

    optimizer.setup(model)

    if 'weight_decay' in config and config['weight_decay'] is not None:
        optimizer.add_hook(chainer.optimizer.WeightDecay(config['weight_decay']))

    if 'gradient_clipping' in config and config['gradient_clipping'] is not None:
        optimizer.add_hook(chainer.optimizer.GradientClipping(config['gradient_clipping']))

    return optimizer


def create_trainer(
        config: TrainConfig,
        project_path: str,
        updater,
        model: typing.Dict,
        eval_func,
        iterator_test,
        iterator_train_eval,
        loss_names,
        converter=chainer.dataset.convert.concat_examples,
        log_name='log.txt',
):
    trainer = chainer.training.Trainer(updater, out=project_path)

    log_trigger = (config.log_iteration, 'iteration')
    save_trigger = (config.save_iteration, 'iteration')

    eval_test_name = 'eval/test'
    eval_train_name = 'eval/train'

    snapshot = extensions.snapshot_object(model['encoder'], 'encoder{.updater.iteration}.model')
    trainer.extend(snapshot, trigger=save_trigger)
    snapshot = extensions.snapshot_object(model['generator'], 'generator{.updater.iteration}.model')
    trainer.extend(snapshot, trigger=save_trigger)
    snapshot = extensions.snapshot_object(model['mismatch_discriminator'], 'mismatch_discriminator{.updater.iteration}.model')
    trainer.extend(snapshot, trigger=save_trigger)

    trainer.extend(utility.chainer.dump_graph([
        'encoder/' + loss_names[0],
        'generator/' + loss_names[0],
        'mismatch_discriminator/' + loss_names[0],
    ], out_name='main.dot'))

    def _make_evaluator(iterator):
        return utility.chainer.NoVariableEvaluator(
            iterator,
            target=model,
            converter=converter,
            eval_func=eval_func,
            device=config.gpu,
        )

    trainer.extend(_make_evaluator(iterator_test), name=eval_test_name, trigger=log_trigger)
    trainer.extend(_make_evaluator(iterator_train_eval), name=eval_train_name, trigger=log_trigger)

    report_target = []
    for evaluator_name in ['', eval_test_name + '/', eval_train_name + '/']:
        for model_name in [s + '/' for s in model.keys()]:
            for loss_name in set(loss_names):
                report_target.append(evaluator_name + model_name + loss_name)

    trainer.extend(extensions.LogReport(trigger=log_trigger, log_name=log_name))
    trainer.extend(extensions.PrintReport(report_target))

    return trainer
