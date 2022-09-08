# sampling methods
import random
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
import time

from tqdm import tqdm



def l1dist(out1, out2):
    return torch.abs(torch.nn.functional.softmax(out1, dim=1) - torch.nn.functional.softmax(out2, dim=1)).sum(1)


def labeled_samples(net, fc, labeled_loader):
    print('calculating...')
    start = time.time()
    net.eval()
    fc.eval()
    all_dist = []
    for images, _, _ in labeled_loader:
        images = images.cuda()

        with torch.no_grad():
            preds, mid = net(images)
            pred_1, pred_2 = fc(mid)

            l1 = l1dist(pred_1, pred_2)
            l11 = l1dist(preds, pred_1)
            l12 = l1dist(preds, pred_2)

            dist = l1 + l11 + l12
            

        dist = dist.cpu().data
        all_dist.extend(dist)

    all_dist = torch.stack(all_dist)
    mean_probs = all_dist.mean()

    finish = time.time()
    print('Sampling time consumed: {:.2f}s'.format(finish - start))

    return mean_probs




def sample(net, fc, unlabeled_loader, mean_probs=0):
    print('sampling...')
    start = time.time()
    net.eval()
    fc.eval()

    all_dist = []
    all_indices = []

    for images, _, indices in unlabeled_loader:
        images = images.cuda()

        with torch.no_grad():
            preds, mid = net(images)
            pred_1, pred_2 = fc(mid)


            l1 = l1dist(pred_1, pred_2)
            l11 = l1dist(preds, pred_1)
            l12 = l1dist(preds, pred_2)

            dist = l1 + l11 + l12

        dist = dist.cpu().data
        all_dist.extend(dist)
        all_indices.extend(indices)

    all_dist = torch.stack(all_dist)
    all_dist = all_dist.view(-1)
    all_dist = all_dist - mean_probs

    _, querry_indices = torch.topk(all_dist, int(2500))
    querry_pool_indices = np.asarray(all_indices)[querry_indices]

    finish = time.time()
    print('Sampling time consumed: {:.2f}s'.format(finish - start))

    return querry_pool_indices
