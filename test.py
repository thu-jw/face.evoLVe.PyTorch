#!/usr/bin/python3
from data.dataset import FaceDataLoader
import os
from tqdm import tqdm

DATA_ROOT = '/home/zwl/data/faces_emore'
train_loader = FaceDataLoader(os.path.join(DATA_ROOT, 'train.rec'))
for data, labels in tqdm(iter(train_loader)):
    print(data, labels)
    print(data.shape, labels.shape)
    break
