import os

from os.path import dirname, join, basename, isfile
from tqdm import tqdm

from models import SyncNet_color as SyncNet
from models import Wav2Lip, Wav2Lip_disc_qual, FeatureExtractor
import audio

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils import data as data_utils
import numpy as np

from glob import glob

import os, random, cv2, argparse
from hparams import hparams, get_image_list

logloss = nn.BCELoss()

def cosine_loss(a, v, y):
    d = nn.functional.cosine_similarity(a, v)
    loss = logloss(d.unsqueeze(1), y)

    return loss
    
def get_sync_loss(mel, g):
    g = g[:, :, :, g.size(3)//2:]
    g = torch.cat([g[:, :, i] for i in range(syncnet_T)], dim=1)
    # B, 3 * T, H//2, W
    a, v = syncnet(mel, g)
    y = torch.ones(g.size(0), 1).float().to(device)
    return cosine_loss(a, v, y)
    
recon_loss = nn.L1Loss()
feature_extractor = FeatureExtractor()
feature_extractor.eval()

# --------- Add content loss here ---------------
def get_content_loss(g, gt):
    
    gen_feautres = feature_extractor(g)
    real_features = feature_extractor(gt)
    loss_content = recon_loss(gen_feautres, real_features.detach())

    return loss_content


def eval_model(test_data_loader, wav2lip_rife, device, eval_steps=300):
    print('Evaluating for {} steps'.format(eval_steps))
    running_sync_loss, running_l1_loss, running_content_loss = [], [], []

    while 1:
        for step, (x, xW, indiv_mels, mel, gt) in enumerate((test_data_loader)):

            x = x.to(device)
            xW = xW.to(device)
            mel = mel.to(device)
            indiv_mels = indiv_mels.to(device)
            gt = gt.to(device)

            # RIFE -> Wav2Lip
            g = wav2lip_rife(indiv_mels, x, xW)

            l1loss = recon_loss(g, gt)
            sync_loss = get_sync_loss(mel, g)
            content_loss = get_content_loss(g, gt)

            running_l1_loss.append(l1loss.item())
            running_sync_loss.append(sync_loss.item())
            running_content_loss.append(content_loss.item())

            if step > eval_steps: break

        print('[Test] L1: {}, Sync: {}, Content: {}'.format(sum(running_l1_loss) / len(running_l1_loss),
                                                            sum(running_sync_loss) / len(running_sync_loss),
                                                            sum(running_content_loss) / len(running_content_loss)))
        return sum(running_sync_loss) / len(running_sync_loss)

