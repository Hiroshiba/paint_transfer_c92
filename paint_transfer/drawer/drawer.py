import chainer
import os

from paint_transfer.config import Config
from paint_transfer.forwarder import Forwarder
from paint_transfer.model import create_model
from paint_transfer import utility


class Drawer(object):
    def __init__(self, path_result_directory, gpu):
        config_path = Config.get_config_path(path_result_directory)
        config = Config(config_path)

        self.forwarder = None  # type: Forwarder

        self.path_result_directory = path_result_directory
        self.dataset_config = config.dataset_config
        self.model_config = config.model_config
        self.gpu = gpu
        self.target_iteration = None

    def _get_path_model(self, iteration, model_type: str):
        return os.path.join(self.path_result_directory, '{}{}.model'.format(model_type, iteration))

    def exist_save_model(self, iteration, model_type: str):
        path_model = self._get_path_model(iteration, model_type)
        return os.path.exists(path_model)

    def load_model(self, iteration):
        self.target_iteration = iteration

        models = create_model(self.model_config)
        for model_type in ['encoder', 'generator', 'mismatch_discriminator']:
            if not self.exist_save_model(iteration, model_type):
                print("warning! iteration {iteration} model is not found.".format(iteration=iteration))
                return False

            path_model = self._get_path_model(iteration, model_type)
            model = getattr(models, model_type)

            print("load {} ...".format(path_model))
            chainer.serializers.load_npz(path_model, model)

            if self.gpu >= 0:
                chainer.cuda.get_device(self.gpu).use()
                model.to_gpu(self.gpu)

        self.forwarder = Forwarder(self.model_config, models=models)
        return True

    def draw(self, input, raw_line):
        if self.gpu >= 0:
            input = chainer.cuda.to_gpu(input)
            raw_line = chainer.cuda.to_gpu(raw_line)

        output = self.forwarder(input=input, raw_line=raw_line)
        output = chainer.cuda.to_cpu(output.data)
        return utility.image.array_to_image(output)
