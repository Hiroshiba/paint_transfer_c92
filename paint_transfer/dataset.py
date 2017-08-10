import chainer
import multiprocessing
import json
import os
import typing

from paint_transfer.config import DatasetConfig
from paint_transfer.data_process import *


class DataProcessDataset(chainer.dataset.DatasetMixin):
    def __init__(self, data: typing.List, data_process: BaseDataProcess, test):
        self._data = data
        self._data_process = data_process
        self._test = test

    def __len__(self):
        return len(self._data)

    def get_example(self, i):
        return self._data_process(data=self._data[i], test=self._test)


T = typing.TypeVar('T')

data_process = ChainProcess([
    PILImageProcess(mode='RGB'),
    RandomFlipImageProcess(p_flip_horizontal=0.5, p_flip_vertical=0),
    RandomResizeImageProcess(min_short=128, max_short=160),
    RandomCropImageProcess(crop_width=128, crop_height=128),
    RgbImageArrayProcess(),
    SplitProcess({
        'target': None,
        'raw_line': RawLineImageArrayProcess(),
    })
])


def data_filter(
        datas: typing.List[T],
        keys: typing.List[str],
        filter_func=typing.Callable[[T], bool],
        num_process: typing.Optional[int] = 1,
        cache_path=None,
):
    assert len(datas) == len(keys)

    if cache_path is not None and os.path.exists(cache_path):
        index = json.load(open(cache_path))
    else:
        index = {}

    unknown_pairs = [(key, data) for key, data in zip(keys, datas) if key not in index]

    with multiprocessing.Pool(num_process) as p:
        unknown_keys = map(lambda kd: kd[0], unknown_pairs)
        unknown_datas = map(lambda kd: kd[1], unknown_pairs)
        filtered = p.map(filter_func, unknown_datas, chunksize=16)

        for key, filt in zip(unknown_keys, filtered):
            index[key] = filt

    # zip -> filter -> unzip -> get data
    filtered_datas = map(lambda kd: kd[1], filter(lambda kd: index[kd[0]], zip(keys, datas)))

    if cache_path is not None:
        json.dump(index, open(cache_path, 'w'))

    return filtered_datas


def filter_image(path, min_size=300, max_size=None, min_aspect=1 / 4, max_aspect=4):
    import numpy
    from PIL import Image
    from skimage.color import rgb2hsv

    try:
        assert os.path.getsize(path) <= 100 * 1000 * 1000, 'file size error'

        image = Image.open(path).convert('RGB')

        w, h = image.size
        if min_size is not None:
            assert w >= min_size and h >= min_size, 'min size error'
        if max_size is not None:
            assert w <= max_size or h <= max_size, 'max size error'

        aspect = w / h
        assert min_aspect is None or aspect >= min_aspect, 'min aspect error'
        assert max_aspect is None or aspect <= max_aspect, 'max aspect error'

        # check data process
        data = data_process(path, test=True)

        # remove low saturation
        rgb = numpy.array(image, numpy.float32) / 255
        hsv = rgb2hsv(rgb)
        sigma = numpy.var(hsv[:, :, 1])
        assert sigma > 0.02, 'saturation error'

    except Exception as e:
        print(path, e)
        return False

    return True


def choose(config: DatasetConfig):
    if config.images_glob is not None:
        import glob
        paths = glob.glob(config.images_glob)
        paths = data_filter(
            datas=paths,
            keys=list(map(lambda p: os.path.basename(p), paths)),
            filter_func=filter_image,
            num_process=None,
            cache_path=config.cache_path,
        )
        paths = list(paths)
    else:
        paths = json.load(open(config.images_list))

    num_test = config.num_test
    train_paths = paths[num_test:]
    test_paths = paths[:num_test]
    train_for_evaluate_paths = train_paths[:num_test]

    return {
        'train': DataProcessDataset(train_paths, data_process, test=False),
        'test': DataProcessDataset(test_paths, data_process, test=True),
        'train_eval': DataProcessDataset(train_for_evaluate_paths, data_process, test=True),
    }
