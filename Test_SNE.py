#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import torch
import os
from torch.nn import DataParallel
import torch

from model import generate_model
from opts import parse_opts
from layers import *
from metrics import *
from DataIter import GGODataIter
from torch.utils.data import DataLoader

import logging
from tqdm import tqdm
from glob import glob 
from torch.nn import functional as F
# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt
def reset_weights(model):
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

if __name__ == '__main__':
    try:
        # Initialize the opts
        opt = parse_opts()
        # make directory
        target_dir = opt.save_dir
        excels_path ='./t-SNE/'
        if not os.path.exists(excels_path):
            os.makedirs(excels_path)
        # 1.logger
        logger = logging.getLogger(__name__)
        logger.setLevel(level=logging.INFO)
        file_handler = logging.FileHandler(
            os.path.join(excels_path, "logInference.txt"))  # 文件句柄
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        console_handler = logging.StreamHandler()  # 流句柄
        console_handler.setLevel(logging.INFO)
        logger.addHandler(console_handler)
        # construct data iterator
        test_path = opt.test_path
        model_path = opt.model_path
        model, policies = generate_model(opt)
        # model.fc = torch.nn.Identity() #去掉最后一层
        if model_path =='random':
            model = DataParallel(model.cuda())
            reset_weights(model)
        else:
            logger.info("Loading pretrained parameters...")
            load_parameters = torch.load(model_path)['state_dict']
            model = DataParallel(model.cuda())
            model.load_state_dict(load_parameters)
        logger.info("Loading successful !!!")

        model.eval()
        logger.info("model finished!!!")
        # testim_list = glob(test_path + '/*.npy')
        
        logger.info("begin classifation")
        train_iter = GGODataIter(
        data_file=test_path,
        phase='test',
        crop_size=opt.sample_size,
        crop_depth=opt.sample_duration,
        aug=opt.aug,
        sample_phase=opt.sample,
        classifier_type=opt.clt)
        
        train_loader = DataLoader(
        train_iter,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=12,
        pin_memory=True)
        
        features = []
        labels = []
        
        logger.info("starting inference...")
        model = model.module
        for i, (data, label, names) in tqdm(enumerate(train_loader)):
            # pred_dict = {}
            # img = np.load(testim, allow_pickle=True)
            # img_r = center_crop(img)
            # imgs = img_r[np.newaxis, :, :, :]
            # imgs = (imgs - 128) / 255.0
            # imgs = imgs[np.newaxis, :, :, :]
            # imgs = torch.from_numpy(imgs.astype(np.float32))
            # data = imgs.cuda(non_blocking=True)
            data = data.cuda(non_blocking=True)
            # label = label.cuda(non_blocking=True)
            if opt.model =='resnet':
                x = model.conv1(data)
                x = model.bn1(x)
                x = model.relu(x)
                x = model.maxpool(x)

                x = model.layer1(x)
                x = model.layer2(x)
                x = model.layer3(x)
                x = model.layer4(x)
                x = model.avgpool(x)
                out = x.view(x.size(0), -1)
            else:#densenet
                features = model.features(x)
                out = F.relu(features, inplace=True)
                out = F.adaptive_avg_pool3d(out,
                        output_size=(1, 1,
                        1)).view(features.size(0), -1)
            # out = model(data)
            # pred = torch.sigmoid(out[:, :])
            # pred_arr = pred.data.cpu().numpy()
            pred_arr = out.data.cpu().numpy()
            
            
            # label_text = testim.split('_')[-2]
            # label = np.zeros((1,),dtype=np.float32)
            # label[0] = 1.0 if label_text == '2' else 0.0
        
        
            features.append(pred_arr)
            labels.append(label)
        logger.info("inference finished!!!")
        features = np.concatenate(features, axis=0)
        labels = np.concatenate(labels, axis=0)

        np.save(
            os.path.join(excels_path, 
                         f'features_{target_dir}.npy'), 
            features)
        np.save(
            os.path.join(excels_path, 
                         f'labels_{target_dir}.npy'), 
            labels)
    except Exception as e:
        logger.error("some bad happened :{}".format(e))
        import pdb
        pdb.set_trace()
