import os
import torch
from torch import nn, optim
from torch.nn import Conv2d, AdaptiveAvgPool2d, Linear
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torchvision import models
from efficientnet_pytorch import EfficientNet


class BaseNet:
    def __init__(self, config):
        self.config = config

        self._build()

        self.model = self.model.cuda()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['learning_rate'], weight_decay=config['wd'])
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

    def _build(self):
        raise NotImplementedError()

    def save(self, epoch, itr):
        snapshot = {
            'epoch': epoch,
            'itr': itr,
            'batch_size': self.config['batch_size'],
            'image_size': self.config['image_size'],
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }

        torch.save(snapshot, os.path.join(self.config['snapshot_dir'], f'epoch_{epoch:04}_itr_{itr:04}.pt'))

    def load(self, path):
        state = torch.load(path)
        epoch = state['epoch']
        itr = state['itr']
        self.config['batch_size'] = state['batch_size']
        self.config['image_size'] = state['image_size']
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


# https://pytorch.org/docs/stable/torchvision/models.html#torchvision.models.resnet18
class ResNet18(BaseNet):
    def __init__(self, config):
        super.__init__(config)

    def _build(self):
        self.model = models.resnet18(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, self.config['class_num'])


class ResNet50(BaseNet):
    def __init__(self, config):
        super.__init__(config)

    def _build(self):
        self.model = models.resnet50(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, self.config['class_num'])


class MobileNetV2(BaseNet):
    def __init__(self, config):
        super.__init__(config)

    def _build(self):
        self.model = models.mobilenet_v2(pretrained=True)
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(num_ftrs, self.config['class_num'])


class EffNet(BaseNet):
    def __init__(self, config):
        super.__init__(config)

    def _build(self):
        self.model = EfficientNet.from_pretrained('efficientnet-b1', num_classes=self.config['class_num'])
        self.model = self.model.cuda()
