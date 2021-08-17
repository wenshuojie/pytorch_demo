# 使用`Dataset`提供数据集的封装
# 使用`Dataloader`实现数据并行加载
# 训练集：数据增强  验证集、测试集：不需要

import os
from PIL import Image
import numpy as np
from torchvision import transforms as T
from torch.utils import data

class DogCat(data.Dataset):

    def __init__(self,root,transforms=None,train=True,test=False):
        # root:
        # E:/pytorch_workspace/Hands-on-learning-and-deep-learning/05_Practice/data/test1/666.jpg
        # E:/pytorch_workspace/Hands-on-learning-and-deep-learning/05_Practice/data/train/cat.666.jpg
        imgs=[os.path.join(root,img) for img in os.listdir(root)]

        # 确定测试集和训练集路径
        self.test = test
        if self.test:
            imgs = sorted(imgs,key=lambda x: int(x.split('.')[-2].split('/')[-1]))
        else:
            imgs = sorted(imgs,key=lambda x:int(x.split('.')[-2]))

        imgs_num = len(imgs)

        # 划分测试集，练集和验证集
        if self.test:
            self.imgs = imgs
        elif train:
            self.imgs = imgs[:int(0.7*imgs_num)]
        else:
            self.imgs = imgs[int(0.7 * imgs_num):]

        # 设置默认transform
        if transforms is None:
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            # 测试集，练集和验证集transform不同
            if self.test or not train:
                self.transforms = T.Compose([
                    T.Resize(224),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    normalize
                ])
            else:
                self.transforms = T.Compose([
                    T.Resize(256),
                    T.RandomReSizedCrop(224),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    normalize
                ])

    def __getitem__(self, index):
        # 测试集，没有label，返回图片id
        img_path = self.imgs[index]
        if self.test:
            label = int(self.imgs[index].split('.')[-2].split('/')[-1])
        else:
            label = 1 if 'dog' in self.imgs[index].split('/')[-1] else 0

        data = Image.open(img_path)
        data = self.transforms(data)
        return data,label

    def __len__(self):
        return len(self.imgs)