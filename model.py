import os
import torch
from torch import nn, optim
from torch.nn import Conv2d, AdaptiveAvgPool2d, Linear
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torchvision import models


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = Conv2d(3, 64, (3, 3), (1, 1), (1, 1))
        self.conv2 = Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.fc = Linear(64, 1049)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = AdaptiveAvgPool2d(1)(x).squeeze()
        x = self.fc(x)
        return x


class WideResNet50:
    def __init__(self, config):
        self.config = config
        self.model = models.wide_resnet50_2(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, config['class_num'])
        self.model = self.model.cuda()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['learning_rate'], weight_decay=config['wd'])
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

    def save(self, epoch, itr):
        snapshot = {
            'epoch': epoch,
            'itr': itr,
            'batch_size': self.config['batch_size'],
            'image_size': self.config['args.size'],
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }

        torch.save(snapshot, os.path.join(self.config['snapshot_dir'], f'epoch_{epoch:04}_itr_{itr:04}.pt'))

    def load(self, path):
        state = torch.load(path)
        epoch = state['epoch']
        itr = state['itr']
        self.args.batch_size = state['batch_size']
        self.args.size = state['size']
        self.model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1, last_epoch=epoch)

        print(f'Loaded from epoch: {epoch}, iter: {itr}')

        return epoch, itr

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)
