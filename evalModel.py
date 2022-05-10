# -*- coding: utf-8 -*-
# @Author : Young
# @Time : 8:36  2022-05-10
import gc
import random
import numpy as np
import torch.nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from dataLoader import RotNetLoader, RotateImage, collate_fn, CenterCrop, largest_rotated_rect
from torch.utils.data import DataLoader

if __name__ == '__main__':
    ptPath = "rotenet_model.pt"
    model = torch.load(ptPath)
    model.to("cpu")
    # dataset
    testdata = RotNetLoader(dirName=r"D:\MyNAS\RotNet-master\data\test", imagesize=512, onehot=True)
    testloader = DataLoader(testdata, batch_size=1, shuffle=True, drop_last=True,
                             num_workers=0, collate_fn=collate_fn)

    selectIndex = random.choices(np.arange(0, len(testdata)), k=len(testdata))
    num = 0
    for i in selectIndex:
        batch = testdata.__getitem__(i)
        img, label = batch
        img = torch.from_numpy(np.array(img[None, ...], dtype=np.float))
        pred = model(img.float())
        pred = F.softmax(pred)
        pred_angle = torch.argmax(pred)
        if abs(pred_angle.item() - label) <= 5:
            print(pred_angle.item())
            num += 1
        else:
            print(pred_angle.item(), label)
    print(num / len(testdata))
    '''
        # show and save rote result
        plt.figure(figsize=(5, 2.3))

        title_fontdict = {
            'fontsize': 10,
            'fontweight': 'bold'
        }

        img1 = img.numpy()[0].transpose(1, 2, 0)*255
        h, w = img1.shape[:2]
        original_image = RotateImage(img1, -label.item())
        original_image = CenterCrop(original_image, *largest_rotated_rect(w, h, -label.item()))

        corrected_image = RotateImage(img1, -pred_angle.item())
        corrected_image = CenterCrop(corrected_image, *largest_rotated_rect(w, h, -pred_angle.item()))

        ax = plt.subplot(1, 3, 1)
        plt.title('Original\n', fontdict=title_fontdict)
        plt.imshow(np.squeeze(original_image).astype('uint8'), **{})
        plt.axis('off')

        ax = plt.subplot(1, 3, 2)
        plt.title('Rotated\n', fontdict=title_fontdict)
        ax.text(0.5, 1.03, 'Angle: {0}'.format(label.item()), horizontalalignment='center',
                transform=ax.transAxes, fontsize=11)
        plt.imshow(img1.astype('uint8'))
        plt.axis('off')

        ax =plt.subplot(1, 3, 3)
        plt.title('Corrected\n', fontdict=title_fontdict)
        ax.text(0.5, 1.03, 'Angle: {0}'.format(pred_angle.item()), horizontalalignment='center',
                transform=ax.transAxes, fontsize=11)

        plt.imshow(corrected_image.astype('uint8'), **{})
        plt.axis('off')

        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.savefig("test_{}.png".format(i))
    '''