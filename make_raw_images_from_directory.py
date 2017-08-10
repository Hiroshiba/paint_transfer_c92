import argparse
import glob
import numpy
import os
from PIL import Image
import sys

from paint_transfer.config import Config
from paint_transfer.data_process import BaseDataProcess
from paint_transfer import dataset
from paint_transfer.utility.image import array_to_image, save_images, save_tiled_image

parser = argparse.ArgumentParser()
parser.add_argument('--config_json_path')
parser.add_argument('--image_directory', required=True)
parser.add_argument('--path_save_directory')
args = parser.parse_args()

config_json_path = args.config_json_path
path_save = args.path_save_directory
os.makedirs(path_save, exist_ok=True)

config = Config(config_json_path)


class CropTopImageSquareProcess(BaseDataProcess):
    def __call__(self, image: Image.Image, test):
        width = image.size[0]
        image = image.crop((0, 0, width, width))
        return image


dataset.data_process._process.insert(3, CropTopImageSquareProcess())

paths = glob.glob(os.path.join(args.image_directory, '*'))
num_image = len(paths)
dataset_test = dataset.DataProcessDataset(paths, data_process=dataset.data_process, test=True)

# save images
images = []
raw_lines = []
for i, data in enumerate(dataset_test[:num_image]):
    image = data['target'][numpy.newaxis]
    image = array_to_image(image)
    images += image

    raw_line = data['raw_line'][numpy.newaxis]
    raw_line = 1 - raw_line
    raw_line = numpy.repeat(raw_line, 3, axis=1)
    raw_line = array_to_image(raw_line, minmax=(0, 1))
    raw_lines += raw_line

save_images(images, path_save, '')
save_images(raw_lines, path_save, 'raw_line_')

# save tiled images
paths_input = [os.path.join(path_save, '{}.png'.format(i)) for i in range(num_image)]
save_tiled_image(paths_input, os.path.join(path_save, 'tile{}.png'.format(num_image)), col=num_image)

paths_input = [os.path.join(path_save, 'raw_line_{}.png'.format(i)) for i in range(num_image)]
save_tiled_image(paths_input, os.path.join(path_save, 'tile_raw_line_{}.png'.format(num_image)), col=num_image)
