import argparse
import glob
import numpy
import os
import sys

from paint_transfer.config import Config
from paint_transfer import dataset
from paint_transfer.utility.image import array_to_image, save_images, save_tiled_image

parser = argparse.ArgumentParser()
parser.add_argument('--config_json_path')
parser.add_argument('--path_save_directory')
parser.add_argument('--num_image', type=int, default=100)
args = parser.parse_args()

config_json_path = args.config_json_path
path_save_directory = args.path_save_directory
num_image = args.num_image

config = Config(config_json_path)

os.path.exists(path_save_directory) or os.mkdir(path_save_directory)

path_save = os.path.join(path_save_directory, 'test')
os.path.exists(path_save) or os.mkdir(path_save)

dataset_test = dataset.choose(config.dataset_config)['test']

# save images
images = []
for i, data in enumerate(dataset_test[:num_image]):
    image = data['target'][numpy.newaxis]
    image = array_to_image(image)
    images += image

save_images(images, path_save, '')

# save tiled images
for num_tile_image in range(10, 110, 10):
    paths_input = [os.path.join(path_save, '{}.png'.format(i)) for i in range(num_tile_image)]
    save_tiled_image(paths_input, os.path.join(path_save, 'tile{}.png'.format(num_tile_image)), col=num_tile_image)
