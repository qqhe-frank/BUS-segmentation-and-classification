import random

import torch
import torchvision.transforms.functional as F
from PIL import Image




class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask, label):
        assert img.size == mask.size
        for t in self.transforms:
            img, mask, label = t(img, mask, label)
        return img, mask, label

class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img, mask, label):
        rotate_degree = random.random() * 2 * self.degree - self.degree
        return img.rotate(rotate_degree, Image.BILINEAR), \
               mask.rotate(rotate_degree, Image.NEAREST), label


class Resize(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask, label):

        img  = F.resize(img, self.size, F.InterpolationMode.BILINEAR)
        mask = F.resize(mask, self.size, F.InterpolationMode.NEAREST)

        return img, mask, label

class RandomHorizontallyFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, mask, label):
        if torch.rand(1) < self.p:
            return F.hflip(img), F.hflip(mask), label
        return img, mask, label

class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, mask, label):
        if torch.rand(1) < self.p:
            return F.vflip(img), F.vflip(mask), label
        return img, mask, label








#
# if __name__ == '__main__':
#     x1 = torch.rand(2, 3, 512, 428)
#     x2 = torch.rand(2, 1, 512, 428)
#     x3 = 1
#     re = Resize([256, 256])
#     y1, y2, y3 = re(x1, x2, x3)
#     print('Output y1 shape:', y1.shape)
#     print('Output y2 shape:', y2.shape)
#     print('Output y3 :', y3)
#



