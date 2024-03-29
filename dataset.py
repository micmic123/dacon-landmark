import numpy as np
import os
import random
import h5py
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


#
# https://docs.python.org/ko/3/howto/unicode.html#comparing-strings
# https://python.flowdas.com/library/unicodedata.html#unicodedata.normalize
# glob() or os.walk()로 얻은 경로 string은 NFD 형식이고 (한글 char의 길이=3)
# 일반적으로 "한글"처럼 얻은 string은 NFC 형식 (한글 char의 길이=1)

class TrainDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.img_path = os.path.join(self.data_dir, 'train.h5')
        self.label_path = os.path.join(self.data_dir, 'train_label.h5')
        self.transform = transform

        self.imgs = h5py.File(self.img_path, 'r')['img']
        self.labels = torch.LongTensor(np.asarray(h5py.File(self.label_path, 'r')['label']))

        self.reverse_index = dict()

        # TODO
        # try to avoid for loop
        for i, v in enumerate(self.labels) :
            v = v.tolist()
            if v in self.reverse_index :
                self.reverse_index[v].append(i)
            else :
                self.reverse_index[v] = [i]

        print(f'Trainset: {len(self.imgs)} images')

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        label = self.labels[idx].tolist()
        two_idx = random.sample(self.reverse_index[label],2)
        image0 = self.imgs[two_idx[0]]
        image1 = self.imgs[two_idx[1]]

        if self.transform:
            image0 = self.transform(image0)
            image1 = self.transform(image1)

        image0 = torch.unsqueeze(image0, 0)
        image1 = torch.unsqueeze(image1, 0)

        image = torch.cat((image0,image1), 0)

        return image, label


class TestDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.img_path = os.path.join(self.data_dir, 'test.h5')
        self.id_path = os.path.join(self.data_dir, 'test_id.npy')
        self.transform = transform

        self.imgs = h5py.File(self.img_path, 'r')['img']
        self.ids = np.load(self.id_path)

        print(f'Testset: {len(self.imgs)} images')

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        image = self.imgs[idx]

        if self.transform:
            image = self.transform(image)

        return image, self.ids[idx]


def get_dataloader(args, test=False):
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(args.image_size),
            # transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    test_dataset = TestDataset(args.data_dir, data_transforms['test'])
    test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)

    if not test:
        train_dataset = TrainDataset(args.data_dir, data_transforms['train'])
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        return train_dataloader, test_dataloader

    return None, test_dataloader
