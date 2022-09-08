# train.py
#!/usr/bin/env  python3

""" train network using pytorch

author baiyu
"""

import os
import sys
import argparse
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import random

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.utils.data.sampler as sampler
import torch.utils.data as data

from conf import settings
from utils import get_training_dataloader, get_test_dataloader, WarmUpLR
import logger

from sampling import *
from model import *


def read_data(dataloader, labels=True):
        if labels:
            while True:
                for img, label, indices in dataloader:
                    yield img, label, indices
        else:
            while True:
                for img, _, _ in dataloader:
                    yield img


def discrepancy(out1, out2):
    return torch.mean(torch.abs(out1.softmax(1) - out2.softmax(1)))


def train(epoch):

    start = time.time()
    net.train()
    FC.train()
    labeled_loader = read_data(cifar10_labeled_loader)
    unlabeled_loader = read_data(unlabeled_dataloader)
    for iter_count in range(train_iterations):
        images, labels, _ = next(labeled_loader)
        unlab_images, _, _ = next(unlabeled_loader)
        if epoch <= args.warm:
            warmup_scheduler.step()
            fc_warmup_scheduler.step()

        if args.gpu:
            labels = labels.cuda()
            images = images.cuda()
            unlab_images = unlab_images.cuda()

        optimizer.zero_grad()
        outputs, mid = net(images)

        out_1, out_2 = FC(mid)


        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        loss_1 = loss_function(out_1, labels)
        loss_2 = loss_function(out_2, labels)

        loss_c = loss_1 + loss_2

        optim_fc.zero_grad()
        loss_c.backward()
        optim_fc.step()

        # unlabeled training
        
        _, mid = net(images)
        out_1, out_2 = FC(mid)

        out_u, mid_u = net(unlab_images)
        out_1_u, out_2_u = FC(mid_u)

        loss_1 = loss_function(out_1, labels)
        loss_2 = loss_function(out_2, labels)

        loss_l = loss_1 + loss_2

        loss_u = discrepancy(out_1_u, out_2_u)
        loss_u1 = discrepancy(out_1_u, out_u)
        loss_u2 = discrepancy(out_2_u, out_u)

        loss_aux = loss_u + loss_u1 + loss_u2

        loss_comb = loss_l - loss_aux

        optim_fc.zero_grad()
        loss_comb.backward()
        optim_fc.step()

        n_iter = (epoch - 1) * train_iterations + iter_count + 1

        last_layer = list(net.children())[-1]
        for name, para in last_layer.named_parameters():
            if 'weight' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
            if 'bias' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=iter_count * args.b + len(images),
            total_samples=len(cifar10_labeled_loader.dataset)
        ))

        #update training loss for each iteration
        writer.add_scalar('Train/loss', loss.item(), n_iter)
        writer.add_scalar('Loss L', loss_l, n_iter)
        writer.add_scalar('Loss aux', loss_aux, n_iter)
        writer.add_scalar('Loss Comb', loss_comb, n_iter)

    for name, param in net.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

    finish = time.time()

    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))

@torch.no_grad()
def eval_training(epoch):

    start = time.time()
    net.eval()

    test_loss = 0.0
    correct = 0.0

    for (images, labels) in cifar10_test_loader:

        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()

        outputs, _ = net(images)
        loss = loss_function(outputs, labels)
        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    finish = time.time()
    if args.gpu:
        print('GPU INFO.....')
        print(torch.cuda.memory_summary(), end='')
    print('Evaluating Network.....')
    print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        test_loss / len(cifar10_test_loader.dataset),
        correct.float() / len(cifar10_test_loader.dataset),
        finish - start
    ))
    print()

    writer.add_scalar('Test/Average loss', test_loss / len(cifar10_test_loader.dataset), epoch)
    writer.add_scalar('Test/Accuracy', correct.float() / len(cifar10_test_loader.dataset), epoch)

    return correct.float() / len(cifar10_test_loader.dataset)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-out_path', type=str, default='./test', help='output file_path')
    parser.add_argument('-method', type=str, default='random', help='type of sampling method')
    args = parser.parse_args()

    #data preprocessing:
    num_images = 50000
    initial_budget = 5000
    budget = 2500
    all_indices = set(np.arange(num_images))
    initial_indices = random.sample(all_indices, initial_budget)
    sampler = data.sampler.SubsetRandomSampler(initial_indices)
    current_indices = list(initial_indices)
    num_classes = 10

    # now the iteration numbers
    train_iterations = num_images // args.b

    # train_iterations

    cifar10_labeled_loader = get_training_dataloader(
        settings.CIFAR10_TRAIN_MEAN,
        settings.CIFAR10_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        sampler=sampler
    )

    cifar10_test_loader = get_test_dataloader(
        settings.CIFAR10_TRAIN_MEAN,
        settings.CIFAR10_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True
    )

    splits = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

    loss_function = nn.CrossEntropyLoss()
    logger = logger.Logger(os.path.join(args.out_path, 'scores.txt'))

    for split in splits:
        net = resnet18().cuda()
        FC = FullyConnected(num_classes).cuda()

        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2) #learning rate decay
        iter_per_epoch = len(cifar10_labeled_loader)
        warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)
        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)

        optim_fc = optim.SGD(FC.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        fc_train_scheduler = optim.lr_scheduler.MultiStepLR(optim_fc, milestones=settings.MILESTONES, gamma=0.2)
        fc_warmup_scheduler = WarmUpLR(optim_fc, iter_per_epoch * args.warm)

        #use tensorboard
        if not os.path.exists(settings.LOG_DIR):
            os.mkdir(settings.LOG_DIR)
        writer = SummaryWriter(log_dir=os.path.join(
                settings.LOG_DIR, args.net, settings.TIME_NOW))
        input_tensor = torch.Tensor(1, 3, 32, 32).cuda()
        writer.add_graph(net, input_tensor)

        #create checkpoint folder to save model
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

        unlabeled_indices = np.setdiff1d(list(all_indices), current_indices)
        unlabeled_sampler = data.sampler.SubsetRandomSampler(unlabeled_indices)
        unlabeled_dataloader = get_training_dataloader(
            settings.CIFAR10_TRAIN_MEAN,
            settings.CIFAR10_TRAIN_STD,
            num_workers=4,
            batch_size=args.b,
            sampler=unlabeled_sampler
            )

        best_acc = 0.0
        for epoch in range(1, settings.EPOCH):
            if epoch > args.warm:
                train_scheduler.step(epoch)
                fc_train_scheduler.step(epoch)

            train(epoch)
            acc = eval_training(epoch)

            #start to save best performance model after learning rate decay to 0.01
            if epoch > settings.MILESTONES[1] and best_acc < acc:
                torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='best'))
                best_acc = acc
                continue

            if not epoch % settings.SAVE_EPOCH:
                torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='regular'))

        writer.close()

        mean_probs = labeled_samples(net, FC, cifar10_labeled_loader)
        sampled_indices = sample(net, FC, unlabeled_dataloader, mean_probs)
        current_indices = list(current_indices) + list(sampled_indices)
        sampler = data.sampler.SubsetRandomSampler(current_indices)
        cifar10_labeled_loader = get_training_dataloader(
            settings.CIFAR10_TRAIN_MEAN,
            settings.CIFAR10_TRAIN_STD,
            num_workers=4,
            batch_size=args.b,
            sampler=sampler
            )
        logger.write('Final accuracy with {}% of data is: {:.4f}'.format(int(split*100), acc))
