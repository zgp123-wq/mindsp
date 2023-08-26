import random
import mindspore.numpy as mnp
from mindspore import Tensor
import mindspore.dataset.vision as C
from utils.box_list import BoxList
import numpy as np
from PIL import Image


def hflip(img):
    img_array = np.array(img)
    flipped_img_array = np.flip(img_array, axis=1)
    flipped_img = Image.fromarray(flipped_img_array)
    return flipped_img

def resize(img_list, box_list=None, min_size=None, max_size=None):
    assert isinstance(min_size, (int, tuple)), f'The type of min_size_train should be int or tuple, got {type(min_size)}.'
    if isinstance(min_size, tuple):
        min_size = random.randint(min_size[0], min_size[1])

    assert img_list.img.shape[:2] == img_list.ori_size, 'img size error when resizing.'
    h, w = img_list.ori_size

    short_side, long_side = min(w, h), max(w, h)
    if min_size / short_side * long_side > max_size:
        scale = max_size / long_side
    else:
        scale = min_size / short_side

    new_h, new_w = int(scale * h), int(scale * w)
    assert (min(new_h, new_w)) <= min_size and (max(new_h, new_w) <= max_size), 'Scale error when resizing.'

    # Resize image using MindSpore transform
    resize_op = C.Resize((new_h, new_w))
    resized_img = resize_op(Tensor(img_list.img))
    img_list.img = resized_img.asnumpy()
    img_list.resized_size = (new_w, new_h)

    if box_list is None:
        return img_list
    else:
        box_list.resize(new_size=(new_w, new_h))

    return img_list, box_list

def random_flip(img_list, box_list, h_prob=0.5, v_prob=None):
    if h_prob and random.random() < h_prob:
        new_img = hflip(img_list.img)
        img_list.img = new_img.asnumpy()
        assert img_list.resized_size == box_list.img_size, 'img size != box size when flipping.'
        box_list.box_flip(method='h_flip')
    if v_prob and random.random() < v_prob:
        raise NotImplementedError('Vertical flip has not been implemented.')

    return img_list, box_list

def to_tensor(img_list):
    new_img = Tensor(img_list.img, mnp.float32)
    img_list.img = new_img
    return img_list

def normalize(img_list, mean=(102.9801, 115.9465, 122.7717), std=(1., 1., 1.)):
    mean = Tensor(mean, mnp.float32)
    std = Tensor(std, mnp.float32)
    img_list.img = (img_list.img - mean) / std
    return img_list

def train_aug(img_list, box_list, cfg):
    img_list, box_list = resize(img_list, box_list, min_size=cfg.min_size_train, max_size=cfg.max_size_train)
    img_list, box_list = random_flip(img_list, box_list, h_prob=0.5)
    img_list = to_tensor(img_list)
    img_list = normalize(img_list)
    return img_list, box_list

def val_aug(img_list, box_list, cfg):
    img_list = resize(img_list, box_list=None, min_size=cfg.min_size_test, max_size=cfg.max_size_test)
    img_list = to_tensor(img_list)
    img_list = normalize(img_list)
    return img_list, None
