# -*- coding:utf-8 -*-
# original author: DuanshengLiu
from Unet import unet_train
from CNN import cnn_train

unet_train()  # Get unet.h5 after training, used for license plate positioning
cnn_train()  # Get cnn.h5 after training for license plate recognition
