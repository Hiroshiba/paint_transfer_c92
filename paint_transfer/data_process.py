from abc import ABCMeta, abstractmethod
import cv2
import numpy
from PIL import Image
from skimage.color import rgb2lab
import typing


class BaseDataProcess(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, data, test):
        pass


class PILImageProcess(BaseDataProcess):
    def __init__(self, mode):
        self._mode = mode

    def __call__(self, path: str, test):
        return Image.open(path).convert(self._mode)


class RandomScaleImageProcess(BaseDataProcess):
    def __init__(self, min_scale: float, max_scale: float):
        self._min_scale = min_scale
        self._max_scale = max_scale

    def __call__(self, image: Image.Image, test):
        base_size = image.size

        rand = numpy.random.rand(1) if not test else 0.5

        scale = rand * (self._max_scale - self._min_scale) + self._min_scale
        size_resize = (int(image.size[0] * scale), int(image.size[1] * scale))

        if base_size != size_resize:
            image = image.resize(size_resize, resample=Image.BICUBIC)

        return image


class RandomResizeImageProcess(BaseDataProcess):
    def __init__(self, min_short: int, max_short: int):
        self._min_short = min_short
        self._max_short = max_short

    def __call__(self, image: Image.Image, test):
        base_size = image.size

        rand = numpy.random.rand(1) if not test else 0.5

        short = rand * (self._max_short - self._min_short + 1) + self._min_short
        short = int(short)
        if short > self._max_short:
            short = self._max_short

        scale = max([short / image.size[0], short / image.size[1]])
        size_resize = (round(image.size[0] * scale), round(image.size[1] * scale))

        if base_size != size_resize:
            image = image.resize(size_resize, resample=Image.BICUBIC)

        return image


class RandomCropImageProcess(BaseDataProcess):
    def __init__(self, crop_width: int, crop_height: int):
        self._crop_width = crop_width
        self._crop_height = crop_height

    def __call__(self, image: Image.Image, test):
        width, height = image.size
        assert width >= self._crop_width and height >= self._crop_height

        if not test:
            top = numpy.random.randint(height - self._crop_height + 1)
            left = numpy.random.randint(width - self._crop_width + 1)
        else:
            top = (height - self._crop_height) // 2
            left = (width - self._crop_width) // 2

        bottom = top + self._crop_height
        right = left + self._crop_width

        image = image.crop((left, top, right, bottom))
        return image


class RandomFlipImageProcess(BaseDataProcess):
    def __init__(self, p_flip_horizontal, p_flip_vertical):
        self._p_flip_horizontal = p_flip_horizontal
        self._p_flip_vertical = p_flip_vertical

    def __call__(self, image: Image.Image, test):
        if not test:
            if numpy.random.rand(1) < self._p_flip_horizontal:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)

            if numpy.random.rand(1) < self._p_flip_vertical:
                image = image.transpose(Image.FLIP_TOP_BOTTOM)

        return image


class RgbImageArrayProcess(BaseDataProcess):
    def __init__(self, normalize=True, dtype=numpy.float32):
        self._normalize = normalize
        self._dtype = dtype

    def __call__(self, image: Image.Image, test):
        image = numpy.asarray(image.convert('RGB'), dtype=self._dtype).transpose(2, 0, 1)
        if self._normalize:
            image = image / 255 * 2 - 1
        return image


class RawLineImageArrayProcess(BaseDataProcess):
    def __call__(self, image: numpy.ndarray, test):
        from scipy import stats

        def dilate_diff(image, range, iterations=1):
            dil = cv2.dilate(image, numpy.ones((range, range), numpy.float32), iterations=iterations)
            image = cv2.absdiff(image, dil)
            return image

        dtype = image.dtype
        rgb = (image.transpose(1, 2, 0) + 1) / 2
        lab = rgb2lab(rgb) / 100

        image = lab[:, :, 0]
        image = dilate_diff(image, 3).astype(numpy.float32)

        rand = 0.2 + (numpy.random.randn(1) / 20 if not test else 0)
        rand = 0.000001 if rand <= 0 else rand
        image = cv2.GaussianBlur(image, (5, 5), rand)

        rand = 0.4 + (numpy.random.randn(1) / 20 if not test else 0)
        rand = 0.000001 if rand <= 0 else rand
        image = cv2.GaussianBlur(image, (5, 5), rand)

        rand = numpy.random.randn(1) / 40 if not test else 0
        image = numpy.power(image, 0.8 + rand)

        image = image.astype(dtype)[numpy.newaxis]
        return image


class ChainProcess(BaseDataProcess):
    def __init__(self, process: typing.Iterable[BaseDataProcess]):
        self._process = process

    def __call__(self, data, test):
        for p in self._process:
            data = p(data, test)
        return data


class SplitProcess(BaseDataProcess):
    def __init__(self, process: typing.Dict[str, typing.Optional[BaseDataProcess]]):
        self._process = process

    def __call__(self, data, test):
        data = {
            k: p(data, test) if p is not None else data
            for k, p in self._process.items()
        }
        return data
