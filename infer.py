import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import cv2
import numpy as np
from torch.utils.data import DataLoader, dataloader

import config
import dataloaders.dataloaders as dataloaders
import models.models as models
import utils.utils as utils
from dataloaders.MPVDataset import MPVDataset
from dataloaders.VitonDataset import VitonDataset
from utils.plotter import evaluate, plot_simple_reconstructions

opt = config.read_arguments(train=False)

#--- create dataloader to populate opt ---#
opt.phase = "test"
dataloaders.get_dataloaders(opt)

dataset_cl = MPVDataset

model = models.OASIS_model(opt)
model = models.put_on_multi_gpus(opt, model)
model.eval()

dataset = dataset_cl(opt, phase=opt.phase)