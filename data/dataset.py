import os
from PIL import Image
from torch.utils import data
import numpy as np
from torchvision import transforms as T

class DogCat(data.Dataset):
    def __init__(self, root, transforms=None, mode=None):
        '''
        目标：获取所有图片地址，并根据训练、验证、测试来划分数据
        mode∈["train","test","val"]
        '''
        assert mode in ["train","test","val"]
        self.mode=mode
        imgs=[os.path.join(root,img) for img in os.listdir(root)]
        if self.mode=="test":
            imgs=sorted(imgs,key=lambda x: int(x.split('.')[-2].split('/')[-1]))
        else:
            imgs=sorted(imgs,key=lambda x: int(x.split('.')[-2]))
        imgs_num=len(imgs)
        # 划分训练、验证集，验证：训练=3：7
        if self.mode=="test":
            self.imgs=imgs
        elif self.mode=="train":
            self.imgs=imgs[:int(0.7*imgs_num)]
        else:
            self.imgs=imgs[int(0.7*imgs_num):]
        if transforms is None:
            # 数据转换操作，测试验证和训练的数据转换有所区别
            normalize=T.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
            # 测试集和验证集 不需要数据增强
            if self.mode=="test" or self.mode=="val":
                self.transforms=T.Compose([T.Scale(224),T.CenterCrop(224),T.ToTensor(),normalize])
            else:
                self.transforms=T.Compose([T.Scale(256),T.RandomResizedCrop(224),T.RandomHorizontalFlip(),T.ToTensor(),normalize])

    def __getitem__(self, index):
        """
        一次返回一张图片的数据
        """
        img_path=self.imgs[index]
        if self.mode=="test":
            label=int(self.imgs[index].split('.')[-2].split('/')[-1])
        else:
            label=1 if 'dog' in img_path.split('/')[-1] else 0
        data=Image.open(img_path)
        data=self.transforms(data)
        return data,label

    def __len__(self):
        return len(self.imgs)