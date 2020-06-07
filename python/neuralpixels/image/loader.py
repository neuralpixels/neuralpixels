
import os
import random
import scipy.io
import threading
import scipy.misc
import numpy as np
import scipy.ndimage
from neuralpixels.image.io import get_img, get_img_paths
from neuralpixels.image.scipy_old import imread, imsave, imresize


def default_epoch_function(epoch_num):
    pass


class ImageLoader(threading.Thread):
    def __init__(
            self,
            image_path,
            width=None,
            height=None,
            batch_size=1,
            random_crop=False,
            random_scale=False,
            dtype=np.float32,
            skip_if_smaller=False,
            on_epoch_fn=default_epoch_function,
            buffer_size=1000
    ):
        super(ImageLoader, self).__init__(daemon=True)

        self.data_ready = threading.Event()
        self.data_copied = threading.Event()
        self.buffer_filled = False

        self.image_path = image_path
        self.width = width
        self.height = height
        self.batch_size = batch_size
        self.random_crop = random_crop
        self.random_scale = random_scale
        self.buffer_size = buffer_size
        self.on_epoch_fn = on_epoch_fn
        self.skip_if_smaller = skip_if_smaller
        self.dtype = dtype

        self.epoch = 0

        self.files = get_img_paths(self.image_path)

        self.img_buffer = np.zeros((buffer_size, self.height, self.width, 3), dtype=np.float32)

        assert len(self.files) > 0, 'Could not find and images in {}'.format(self.image_path)

        self.available = set(range(self.buffer_size))
        self.ready = set()

        self.cwd = os.getcwd()
        self.start()

    def run(self):
        while True:
            random.shuffle(self.files)
            for f in self.files:
                self.add_to_buffer(f)
            self.epoch += 1
            self.on_epoch_fn(self.epoch)

    def add_to_buffer(self, filename):
        try:
            img = get_img(filename)
            if img.shape[0] < self.height or img.shape[1] < self.width:
                if self.skip_if_smaller:
                    self.files.remove(filename)
                    return
                else:
                    # scale up image first to the min dimension
                    height_scalar = self.height / img.shape[0]
                    width_scalar = self.width / img.shape[1]
                    if height_scalar > width_scalar:
                        # use width as min
                        new_width = self.width
                        new_height = int(new_width / img.shape[1] * img.shape[0])
                        new_img_shape = (new_width, new_height, 3)
                    else:
                        # use height as min
                        new_height = self.height
                        new_width = int(new_height / img.shape[0] * img.shape[1])
                        new_img_shape = (new_width, new_height, 3)
                    img = imresize(img, new_img_shape)

            # scale img
            # see if we can scale
            if self.height < img.shape[0] and self.width < img.shape[1]:
                if self.random_scale:
                    height_scalar = self.height / img.shape[0]
                    width_scalar = self.width / img.shape[1]
                    if height_scalar > width_scalar:
                        # use width as min
                        new_width = random.randint(self.width, img.shape[1])
                        new_height = int(new_width / img.shape[1] * img.shape[0])
                        new_img_shape = (new_width, new_height, 3)
                    else:
                        # use height as min
                        new_height = random.randint(self.height, img.shape[0])
                        new_width = int(new_height / img.shape[0] * img.shape[1])
                        new_img_shape = (new_width, new_height, 3)
                    img = imresize(img, new_img_shape)
                else:
                    height_scalar = self.height / img.shape[0]
                    width_scalar = self.width / img.shape[1]
                    if height_scalar > width_scalar:
                        # use width as min
                        new_width = self.width
                        new_height = int(new_width / img.shape[1] * img.shape[0])
                        new_img_shape = (new_width, new_height, 3)
                    else:
                        # use height as min
                        new_height = self.height
                        new_width = int(new_height / img.shape[0] * img.shape[1])
                        new_img_shape = (new_width, new_height, 3)
                    img = imresize(img, new_img_shape)

            # crop
            if self.height < img.shape[0] or self.width < img.shape[1]:
                height_diff = img.shape[0] - self.height
                width_diff = img.shape[1] - self.width

                if self.random_crop:
                    # random crop
                    height_offset = random.randint(0, height_diff) if height_diff > 0 else 0
                    width_offset = random.randint(0, width_diff) if width_diff > 0 else 0
                else:
                    # central crop
                    height_offset = int(height_diff / 2) if height_diff > 0 else 0
                    width_offset = int(width_diff / 2) if width_diff > 0 else 0

                img = img[height_offset:height_offset + self.height, width_offset:width_offset + self.width, :]

        except Exception as e:
            print('Could not load `{}` as image. \n - {}'.format(filename, e))
            self.files.remove(filename)
            return

        while len(self.available) == 0:
            self.data_copied.wait()
            self.data_copied.clear()

        i = self.available.pop()
        self.img_buffer[i] = img
        self.ready.add(i)

        if len(self.ready) >= self.batch_size:
            self.buffer_filled = True
            self.data_ready.set()

    def get_batch(self, np_array=None):
        self.data_ready.wait()
        self.data_ready.clear()

        if np_array is None:
            np_array = np.zeros([self.batch_size, self.height, self.width, 3])

        for i, j in enumerate(random.sample(self.ready, self.batch_size)):
            np_array[i] = self.img_buffer[j]
            self.available.add(j)
        self.data_copied.set()
