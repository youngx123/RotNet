# -*- coding: utf-8 -*-
# @Author : youngx
# @Time : 10:20  2022-05-07
import gc

import numpy as np
import torch.nn

from model.Net import RotNet, RotNetRegression
from model.model import VIT
from dataLoader import RotNetLoader, RotateImage, collate_fn
from torch.utils.data import DataLoader
from utils import angle_regression_loss
import argparse


def trainModel(config):
    if config.modelName == "RoteNet":
        if config.angleNum == 1:
            model = RotNetRegression(imagesize=config.imageize)
            lossFunc = angle_regression_loss
        else:
            model = RotNet(imagesize=config.imageize, angleNum=config.angleNum)
            lossFunc = torch.nn.CrossEntropyLoss()
    else:  # config.modelName == "VIT":
        model = VIT(image_size=config.imageize, patch_size=config.patch_size, num_classes=config.angleNum,
                    dim=config.dim, transLayer=config.transLayer, multiheads=config.multiheads)
        lossFunc = torch.nn.CrossEntropyLoss()

    # dataset
    traindata = RotNetLoader(dirName=r"D:\MyNAS\RotNet-master\data\part1", imagesize=config.imageize, onehot=True)
    trainloader = DataLoader(traindata, batch_size=config.batchsize, shuffle=True, drop_last=True,
                             num_workers=4, collate_fn=collate_fn)
    testdata = RotNetLoader(dirName=r"D:\MyNAS\RotNet-master\data\eval", imagesize=config.imageize, onehot=True)
    testloader = DataLoader(testdata, batch_size=1, shuffle=True, drop_last=True,
                            num_workers=0, collate_fn=collate_fn)

    model = torch.load("rotenet_model.pt")
    # model.load_sgittate_dict(weightDict)
    device = config.device
    model.train()
    model.to(device)
    optzimer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # # eval parame
    eval_step = len(traindata) // 2
    best_loss = np.Inf
    for epoch in range(config.EPOCH):
        for step, batch in enumerate(trainloader):
            img, label = batch
            img = img.to(device).float()
            label = label.to(device).long()

            pred = model(img)

            loss = lossFunc(pred, label)
            optzimer.zero_grad()
            loss.backward()
            optzimer.step()

            if step % 50 == 0:
                print("epoch %d / %d , step %d / %d , loss %.4f" % (
                epoch, config.EPOCH, step, len(trainloader), loss.item()))
            del img, label, batch, pred
            gc.collect()
            torch.cuda.empty_cache()
            if step % eval_step == 0:
                torch.save(model, "rotenet_model.pt")
                # model.eval()
                # best_loss = evalModel(model, lossFunc, best_loss, testloader)

            model.train()

    torch.save(model, "rotenet_model.pt")


def evalModel(net, lossFunc, bestLoss, evalloader):
    net.eval()
    valloss = 0
    for step, evalbatch in enumerate(evalloader):
        valimg, vallabel = evalbatch
        del evalbatch
        gc.collect()
        torch.cuda.empty_cache()

        valimg = valimg.to("cuda").float()
        vallabel = vallabel.to("cuda").long()

        valpred = net(valimg)
        valloss += lossFunc(valpred, vallabel)

    valloss /= len(evalloader)
    print("save model pt")
    del valpred, vallabel, valimg
    gc.collect()
    torch.cuda.empty_cache()
    if bestLoss > valloss:
        print("eval loss improved from {} to {}".format(bestLoss, valloss))
        bestLoss = valloss
        torch.save(net, "rotenet_best.pt")
    else:
        print("eval loss dont improved from {} , eval loss {}".format(bestLoss, valloss))

    del net, evalloader
    gc.collect()
    torch.cuda.empty_cache()
    return bestLoss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelName", type=str, default="VIT", choices=["RoteNet", "VIT"])
    parser.add_argument("--imageize", type=int, default=512)
    parser.add_argument("--EPOCH", type=int, default=100)
    parser.add_argument("--angleNum", type=int, default=360, choices=[1, 360])

    # vit parameters
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--transLayer", type=int, default=6)
    parser.add_argument("--multiheads", type=int, default=4)
    #
    parser.add_argument("--batchsize", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    arg = parser.parse_args()

    trainModel(arg)
