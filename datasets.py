import json
import os
import torch.utils.data as data
from PIL import Image


def make_dataset(root):

    img_num_class = [cla for cla in os.listdir(os.path.join(root, 'imgs'))
                 if os.path.isdir(os.path.join(root, 'imgs', cla))]

    img_num_class.sort()

    class_indices = dict((k, v) for v, k in enumerate(img_num_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=1)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    image_class1 = class_indices['benign']

    a = [(os.path.join(root, 'imgs', 'benign', img_name),
             os.path.join(root, 'masks', 'benign', img_name.split('.')[0] + '_mask.png'),
             image_class1) for img_name in os.listdir(os.path.join(root, 'imgs', 'benign'))]

    image_class2 = class_indices['malignant']

    b = [(os.path.join(root, 'imgs', 'malignant', img_name),
             os.path.join(root, 'masks', 'malignant', img_name.split('.')[0] + '_mask.png'),
             image_class2) for img_name in os.listdir(os.path.join(root, 'imgs', 'malignant'))]
    c = a+b
    return c

class ImageFolder(data.Dataset):
    def __init__(self, root, joint_transform=None, transform=None, target_transform=None):
        self.root  = root
        self.imgs = make_dataset(root)
        self.joint_transform = joint_transform
        self.transform = transform
        self.target_transform = target_transform


    def __getitem__(self, index):
        img_path, gt_path, label = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        target = Image.open(gt_path).convert('L')
        cla = label
        if self.joint_transform is not None:
            img, target, cla = self.joint_transform(img, target, cla)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, cla

    def __len__(self):
        return len(self.imgs)

class ImageFolder2(data.Dataset):
    def __init__(self, root, joint_transform=None, transform=None, target_transform=None):
        self.root  = root
        self.imgs = make_dataset(root)
        self.joint_transform = joint_transform
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img_path, gt_path = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        target = Image.open(gt_path).convert('L')
        if self.joint_transform is not None:
            img, target = self.joint_transform(img, target)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)



