import argparse
import time
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from prefetch_generator import BackgroundGenerator

from sklearn import metrics

from dataset import train_audioset
from models.Encoder import AuT
from train.lr_schedule import step_learning_rate

parser = argparse.ArgumentParser()
parser.add_argument('--train_dictionary', type=str, default='/data/hhd_projects/audio_dataset/data/balanced_train_data')
parser.add_argument('--val_dictionary', type=str, default='/data/hhd_projects/audio_dataset/data/eval_data')

parser.add_argument('--base_lr', type=float, default=1e-5)
parser.add_argument('--weight_decay', type=float, default=0.1)
parser.add_argument('--optimizer_momentum', type=list, default=[0.9, 0.999])
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr_schedule', type=str, default='')
parser.add_argument('--label_smooth', type=float, default=0.1)

parser.add_argument('--warmup_epochs', type=int, default=3)
parser.add_argument('--num_epochs', type=int, default=500)
parser.add_argument('--reduce_n_epoch', type=int, default=10)
parser.add_argument('--reduce_rate', type=float, default=0.2)

if __name__ == "__main__":
    opt = parser.parse_args()
    print('step1 load the data')
    train_dataset = train_audioset.AudioDataset(opt.train_dictionary)
    train_data_loader = data.DataLoader(train_dataset,
                                        batch_size=opt.batch_size,
                                        shuffle=True,
                                        num_workers=8)

    val_dataset = train_audioset.AudioDataset(opt.val_dictionary, train=False)
    val_data_loader = data.DataLoader(val_dataset,
                                      batch_size=opt.batch_size,
                                      shuffle=True,
                                      num_workers=8)
    print('step2 load the model')
    model = AuT(embedding_kernel=16, embedding_stride=4, embedding_depth=6,dim=2048,heads=16,dim_head=128, depth=24)
    model.init_weight()

    criterion = nn.MultiLabelSoftMarginLoss()

    # if cuda model move to cuda
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model = model.cuda()

    optim = torch.optim.Adam(model.parameters(), weight_decay=opt.weight_decay, betas=opt.optimizer_momentum,
                             lr=opt.base_lr)

    writer = SummaryWriter("AuT_train_track")

    lr_adj = 1
    start_epoch = 0
    start_iter = 0
    n_iter = start_iter
    print('step3 start training the model')
    for epoch in range(start_epoch, opt.num_epochs):
        model.train()
        # use prefetch_generator and tqdm for iterating through data
        pbar = tqdm(enumerate(BackgroundGenerator(train_data_loader)),
                    total=len(train_data_loader))
        start_time = time.time()
        total_loss = 0
        train_total_mAP = 0

        for i, data in pbar:
            lr, lr_adj = step_learning_rate(opt, epoch, n_iter, optim, len(pbar), lr_adj, i)
            writer.add_scalar('lr', lr, n_iter)
            wav, label = data
            if use_cuda:
                wav = wav.float().cuda()
                label = label.float().cuda()
                label = torch.where(label > 0.5, 1-opt.label_smooth, opt.label_smooth / 527)
            prepare_time = start_time - time.time()

            y = model(wav)
            loss = criterion(y, label)
            optim.zero_grad()
            loss.backward()
            total_loss += loss
            optim.step()
            writer.add_scalar('Loss/train', loss, n_iter)
            n_iter += 1
            process_time = start_time - time.time() - prepare_time
            pbar.set_description("Compute efficiency: {:.2f}, epoch: {}/{} lr: {} loss:{}".format(
                process_time / (process_time + prepare_time), epoch, opt.num_epochs, optim.param_groups[0]['lr'], loss))
            start_time = time.time()
        if epoch % 5 == 0 and epoch != 0:
            model.eval()
            pbar = tqdm(enumerate(BackgroundGenerator(val_data_loader)),
                        total=len(val_data_loader))
            start_time = time.time()

            # every batch train loop
            for i, data in pbar:
                wav, label = data
                if use_cuda:
                    wav = wav.float().cuda()
                    target = label
                    label = label.float().cuda()
                if i == 0:
                    scores = torch.sigmoid(model(wav)).cpu().detach().numpy()
                    targets = label.cpu().detach().numpy()

                else:
                    scores = np.append(scores, torch.sigmoid(model(wav)).cpu().detach().numpy(), axis=0)
                    targets = np.append(targets, label.cpu().detach().numpy(), axis=0)

            mAP = metrics.average_precision_score(targets, scores)
            writer.add_scalar('val/mAP', mAP, epoch + 1)

        avg_loss = total_loss / len(pbar)
        writer.add_scalar('Loss/train_batch', avg_loss, epoch + 1)
