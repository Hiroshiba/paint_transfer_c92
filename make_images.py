import argparse
import more_itertools
import numpy
import os
import sys

from paint_transfer.config import Config
from paint_transfer import dataset
from paint_transfer.drawer import Drawer
from paint_transfer.utility.image import save_images, save_tiled_image
from paint_transfer import utility

parser = argparse.ArgumentParser()
parser.add_argument('path_result_directory')
parser.add_argument('--target_iteration', type=int, nargs='+')
parser.add_argument('--num_image', type=int, default=10)
parser.add_argument('--batchsize', type=int, default=10)
parser.add_argument('--gpu', type=int, default=-1)
args = parser.parse_args()

path_result_directory = args.path_result_directory

config_path = Config.get_config_path(path_result_directory)
config = Config(config_path)

path_images = os.path.join(path_result_directory, 'make_images')
os.path.exists(path_images) or os.mkdir(path_images)

dataset_test = dataset.choose(config.dataset_config)['test']

drawer = Drawer(path_result_directory=path_result_directory, gpu=args.gpu)

for target_iteration in args.target_iteration:
    assert drawer.load_model(target_iteration)

    path_save = os.path.join(path_images, 'test_{}'.format(target_iteration))
    os.path.exists(path_save) or os.mkdir(path_save)

    # make image
    paths_tile = []
    for i_input in range(args.num_image):
        data_input = dataset_test[i_input]['target'][numpy.newaxis]

        images = []
        for batch in more_itertools.chunked(dataset_test[:args.num_image], args.batchsize):
            input = numpy.repeat(data_input, len(batch), axis=0)
            batch = utility.chainer.concat_recursive(batch)
            images += drawer.draw(input=input, raw_line=batch['raw_line'])

        paths = save_images(images, path_save, 'input{}_'.format(i_input))
        path_tile = save_tiled_image(paths, col=args.num_image)
        paths_tile += [path_tile]
    save_tiled_image(paths_tile, col=1)
