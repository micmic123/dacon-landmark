import pandas as pd
import os
from datetime import datetime
from tqdm import tqdm

import argparse
import torch
from torch import nn
from dataset import get_dataloader
from model import ResNet18, ResNet50, EffNet, RepresentationNet

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
base = "../../datasets/landmark"

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default=base)
parser.add_argument('--test_sample_csv', default=f"{base}/sample_submission.csv")
parser.add_argument('--image_size', dest='image_size', type=int, default=256)
parser.add_argument('--epochs', dest='epochs', type=int, default=100)
parser.add_argument('--learning_rate', dest='learning_rate', type=float, default=0.001)
parser.add_argument('--wd', dest='wd', type=float, default=1e-5)
parser.add_argument('--temperature', dest='temperature', type=float, default=0.5)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--test_batch_size', type=int, default=256)

parser.add_argument('--test', default=False, action='store_true')
parser.add_argument('--resume', default='', help='snapshot path')
parser.add_argument('--snapshot', default='', help='snapshot path')

parser.add_argument('--test_itr', type=int, default=100)
parser.add_argument('--save_itr', type=int, default=100)
args = parser.parse_args()


now = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
output_dir = f'./outputs/{now}'
snapshot_dir = f'./snapshots/{now}'
log_dir = f'./logs/{now}'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(snapshot_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)


train_dataloader, test_dataloader = get_dataloader(args, test=args.test)

import sys

# train
def train(model, epoch, itr):
    model.train()
    for epoch in range(epoch, args.epochs, 1):
        epoch_loss = 0.
        for image, label in train_dataloader:
            image, label = image.cuda(), label.cuda()

            image = image.view([-1]+list(image.shape)[2:])

            pred = model(image)
            pred = pred.view([pred.shape[0]//2,2]+list(pred.shape)[1:])

            loss = model.criterion(pred[:,0],pred[:,1])

            model.optimizer.zero_grad()
            loss.backward()
            model.optimizer.step()
            epoch_loss += loss.detach().item()
            print(f'epoch : {epoch} step : [{itr}/{len(train_dataloader)}] loss : {loss.detach().item()}')
            itr += 1

            if itr % args.save_itr == 0:
                model.save(epoch, itr)

            if itr % args.test_itr == 0:
                test(model, f'result_epoch_0{epoch + 1:03}_itr_{itr + 1:04}.csv.csv')
                model.train()

            if itr == len(train_dataloader):
                itr = 0
                break

        epoch_loss /= len(train_dataloader)
        model.scheduler.step()
        print('\nepoch : {0} epoch loss : {1}\n'.format(epoch, epoch_loss))


# test
def test(model, filename='final_result.csv'):
    print('test...')
    model.eval()
    submission = None
    dfs = []

    for images, ids in tqdm(test_dataloader, total=len(test_dataloader)):
        images = images.cuda()

        with torch.no_grad():
            preds = model(images)
            preds = nn.Softmax(dim=1)(preds)
            preds = preds.detach().cpu()
            landmark_ids = torch.argmax(preds, dim=1)
        confidences = preds[range(preds.shape[0]), landmark_ids]
        df = {
            'id': ids,
            'landmark_id': landmark_ids.tolist(),
            'conf': confidences.tolist()
        }
        dfs.append(pd.DataFrame(data=df))

    submission = pd.concat(dfs, ignore_index=True)
    path = os.path.join(output_dir, filename)
    submission.to_csv(path, index=False)

def train_representation(model, epoch, itr):
    model.train()
    for epoch in range(epoch, args.epochs, 1):
        epoch_loss = 0.
        for image, label in train_dataloader:
            image, label = image.cuda(), label.cuda()

            pred = model(image)
            loss = model.criterion(input=pred, target=label)

            model.optimizer.zero_grad()
            loss.backward()
            model.optimizer.step()
            epoch_loss += loss.detach().item()
            print(f'epoch : {epoch} step : [{itr}/{len(train_dataloader)}] loss : {loss.detach().item()}')
            itr += 1

            if itr % args.save_itr == 0:
                model.save(epoch, itr)

            if itr % args.test_itr == 0:
                test(model, f'result_epoch_0{epoch + 1:03}_itr_{itr + 1:04}.csv.csv')
                model.train()

            if itr == len(train_dataloader):
                itr = 0
                break

        epoch_loss /= len(train_dataloader)
        model.scheduler.step()
        print('\nepoch : {0} epoch loss : {1}\n'.format(epoch, epoch_loss))

config = {
    'class_num': 1049,
    'learning_rate': args.learning_rate,
    'wd': args.wd,
    'snapshot_dir': snapshot_dir,
    'batch_size': args.batch_size,
    'image_size': args.image_size,
    'temperature': args.temperature,
    'with_fc': False
}
# front_model = MobileNetV2(config)
# front_model = ResNet18(config)
# front_model = ResNet50(config)
front_model = EffNet(config)

model = RepresentationNet(front_model.model, config, 1280, 512, 128)

print(model.__dict__)

if not args.test:
    epoch, itr = 0, 0
    if args.resume:
        epoch, itr = model.load(args.resume)

    train(model, epoch, itr)
    test(model)
else:
    model.load(args.snapshot)
    test(model)
