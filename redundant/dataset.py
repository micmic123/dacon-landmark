import pandas as pd
import os
import unicodedata
from glob import glob
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class TrainDataset(Dataset):
    def __init__(self, args, transfrom):
        self.train_dir = args.train_dir
        self.train_csv_dir = args.train_csv_dir
        self.train_csv_exist_path = args.train_csv_exist_path
        self.args = args
        self.train_image = []
        self.train_label = []
        self.transform = transfrom

        if not os.path.isfile(self.train_csv_exist_path):
            self.train_csv = pd.read_csv(self.train_csv_dir)
            self.train_csv_exist = pd.DataFrame(data=[], columns=['path', 'label'])
            self.load_full_data()
            self.train_csv_exist.to_csv(self.train_csv_exist_path, index=False)
        else:
            self.load_exist_data()
        print(f'Trainset: {len(self.train_image)} images')

    def load_full_data(self):
        # https://docs.python.org/ko/3/howto/unicode.html#comparing-strings
        # https://python.flowdas.com/library/unicodedata.html#unicodedata.normalize
        # glob() or os.walk()로 얻은 경로 string은 NFD 형식이고 (한글 char의 길이=3)
        # 일반적으로 "한글"처럼 얻은 string은 NFC 형식 (한글 char의 길이=1)
        def NFC(s):
            return unicodedata.normalize('NFC', s)

        id2label = dict()
        for i, row in self.train_csv.iterrows():
            id2label[NFC(row['id'])] = row['landmark_id']

        dfs = []
        paths = glob(os.path.join(self.train_dir, "*/*/*"))
        for path in tqdm(paths):
            _id = os.path.splitext(os.path.basename(path))[0]
            _label = id2label[NFC(_id)]
            _row = pd.DataFrame(data=[[path, _label]], columns=['path', 'label'])
            dfs.append(_row)
            self.train_image.append(path)
            self.train_label.append(_label)

        self.train_csv_exist = pd.concat(dfs, ignore_index=True)

    def load_exist_data(self):
        self.train_csv_exist = pd.read_csv(self.train_csv_exist_path)
        for i in tqdm(range(len(self.train_csv_exist)), desc='Loading trainset', total=len(self.train_csv_exist)):
            path = self.train_csv_exist['path'][i]
            label = self.train_csv_exist['label'][i]
            self.train_image.append(path)
            self.train_label.append(label)
        print(len(self.train_image))

    def __len__(self):
        return len(self.train_image)

    def __getitem__(self, idx):
        image = Image.open(self.train_image[idx])
        image = self.transform(image)
        label = self.train_label[idx]

        return image, label


class TestDataset(Dataset):
    def __init__(self, args, transfrom):
        self.test_dir = args.test_dir
        self.test_csv_dir = args.test_csv_dir
        self.test_csv_exist_path = args.test_csv_exist_path
        self.args = args
        self.test_image = []
        self.ids = []
        self.transform = transfrom

        if not os.path.isfile(self.test_csv_exist_path):
            self.test_csv = pd.read_csv(self.test_csv_dir)
            self.test_csv_exist = pd.DataFrame(columns=['path', 'id', 'landmark_id', 'conf'])
            self.load_full_data()
            self.test_csv_exist.to_csv(self.test_csv_exist_path, index=False)
        else:
            self.load_exist_data()
        print(f'Testset: {len(self.test_image)} images')

    def load_full_data(self):
        id2path = dict()
        paths = glob(os.path.join(self.test_dir, "*/*"))
        for path in paths:
            _id = os.path.splitext(os.path.basename(path))[0]
            id2path[_id] = path

        dfs = []
        for i, row in self.test_csv.iterrows():
            _id = row['id']
            path = id2path[_id]
            dfs.append(pd.DataFrame(data=[[path, _id, 0, 0]], columns=['path', 'id', 'landmark_id', 'conf']))
            self.test_image.append(path)
            self.ids.append(_id)

        self.test_csv_exist = pd.concat(dfs, ignore_index=True)

    def load_exist_data(self):
        self.test_csv_exist = pd.read_csv(self.test_csv_exist_path)
        for i in tqdm(range(len(self.test_csv_exist)), desc='Loading testset', total=len(self.test_csv_exist)):
            path = self.test_csv_exist['path'][i]
            _id = self.test_csv_exist['id'][i]
            self.test_image.append(path)
            self.ids.append(_id)

    def __len__(self):
        return len(self.test_image)

    def __getitem__(self, idx):
        image = Image.open(self.test_image[idx])
        image = self.transform(image)

        return image, self.ids[idx]


def get_dataloader(args):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(args.image_size),
            # transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Dataset, Dataloader 정의
    train_dataset = TrainDataset(args, data_transforms['train'])
    test_dataset = TestDataset(args, data_transforms['test'])
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)

    return train_dataloader, test_dataloader
