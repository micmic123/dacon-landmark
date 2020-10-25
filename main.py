import pandas as pd
import os
from datetime import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import torch
from torch import nn
from dataset import get_dataloader
from model import WideResNet50

base = "/content/drive/My Drive/datasets/landmark_kr"

parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', default=f"{base}/public/train/")
parser.add_argument('--train_csv_dir', default=f"{base}/public/train.csv")
parser.add_argument('--train_csv_exist_path', default=f"{base}/public/train_exist.csv")
parser.add_argument('--test_dir', default=f"{base}/public/test/")
parser.add_argument('--test_csv_dir', default=f"{base}/public/sample_submission.csv")
parser.add_argument('--test_csv_exist_path', default=f"{base}/public/sample_submission_exist.csv")

parser.add_argument('--image_size', dest='image_size', type=int, default=256)
parser.add_argument('--epochs', dest='epochs', type=int, default=100)
parser.add_argument('--learning_rate', dest='learning_rate', type=float, default=0.001)
parser.add_argument('--wd', dest='wd', type=float, default=1e-5)
parser.add_argument('--batch_size', dest='batch_size', type=int, default=64)

parser.add_argument('--train', dest='train', type=bool, default=True)
parser.add_argument('--snapshot', default="./snapshots/asdf.pt")

parser.add_argument('--test_iter', type=int, default=50)
parser.add_argument('--save_iter', type=int, default=50)
args = parser.parse_args()


now = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
output_dir = f'./outputs/{now}'
snapshot_dir = f'./snapshots/{now}'
log_dir = f'./logs/{now}'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(snapshot_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)


train_dataloader, test_dataloader = get_dataloader(args)


# train
def train(model, epoch, itr):
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
                test(model, os.path.join(args.test_csv_submission_dir, f'result_epoch_0{epoch + 1:03}.csv'))

            if itr == len(train_dataloader):
                itr = 0

        epoch_loss /= len(train_dataloader)
        model.scheduler.step()
        print('\nepoch : {0} epoch loss : {1}\n'.format(epoch, epoch_loss))

        if (epoch + 1) % args.test_period == 0:
            test(model, f'result_epoch_0{epoch + 1:03}.csv')


# test
def test(model, filename='final_result.csv'):
    print('test...')
    model.eval()
    submission = None
    dfs = []

    for images, ids in test_dataloader:
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


config = {
    'class_num': 1049,
    'learning_rate': args.learning_rate,
    'wd': args.wd,
    'snapshot_dir': snapshot_dir,
    'batch_size': args.batch_size,
    'image_size': args.image_size
}
model = WideResNet50(config)


if args.train:
    epoch, itr = 0, 0
    if args.resume:
        epoch, itr = model.load(args.resume)

    train(model, epoch, itr)
    test(model)
else:
    model.load(args.snapshot)
    test(model)
