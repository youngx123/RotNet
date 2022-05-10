# -*- coding: utf-8 -*-
# @Author : youngx
# @Time : 9:40  2022-05-07
from torch.utils.data import Dataset
import torch.nn as nn
import torch
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

def RotateImage(image, angle):
    """
    Rotates an OpenCV 2 / NumPy image about it's centre by the given angle
    (in degrees). The returned image will be large enough to hold the entire
    new image, with a black background
    Source: http://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
    """
    # Get the image size
    image_size = (image.shape[1], image.shape[0])
    image_center = tuple(np.array(image_size) / 2)

    # Convert the OpenCV 3x2 rotation matrix to 3x3
    rot_mat = np.vstack([cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]])

    rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

    # Shorthand for below calcs
    image_w2 = image_size[0] * 0.5
    image_h2 = image_size[1] * 0.5

    # Obtain the rotated coordinates of the image corners
    rotated_coords = [
        (np.array([-image_w2, image_h2]) * rot_mat_notranslate).A[0],
        (np.array([image_w2, image_h2]) * rot_mat_notranslate).A[0],
        (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
        (np.array([image_w2, -image_h2]) * rot_mat_notranslate).A[0]
    ]

    # Find the size of the new image
    x_coords = [pt[0] for pt in rotated_coords]
    x_pos = [x for x in x_coords if x > 0]
    x_neg = [x for x in x_coords if x < 0]

    y_coords = [pt[1] for pt in rotated_coords]
    y_pos = [y for y in y_coords if y > 0]
    y_neg = [y for y in y_coords if y < 0]

    right_bound = max(x_pos)
    left_bound = min(x_neg)
    top_bound = max(y_pos)
    bot_bound = min(y_neg)

    new_w = int(abs(right_bound - left_bound))
    new_h = int(abs(top_bound - bot_bound))

    # We require a translation matrix to keep the image centred
    trans_mat = np.matrix(
        [[1, 0, int(new_w * 0.5 - image_w2)],
         [0, 1, int(new_h * 0.5 - image_h2)],
         [0, 0, 1]]
    )

    # Compute the tranform for the combined rotation and translation
    affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

    # Apply the transform
    result = cv2.warpAffine(image, affine_mat, (new_w, new_h), flags=cv2.INTER_LINEAR)

    return result


def largest_rotated_rect(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle within the rotated rectangle.

    Original JS code by 'Andri' and Magnus Hoff from Stack Overflow

    Converted to Python by Aaron Snoswell

    Source: http://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
    """

    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return (bb_w - 2 * x, bb_h - 2 * y)


def CenterCrop(image, width, height):
    """
    Given a NumPy / OpenCV 2 image, crops it to the given width and height,
    around it's centre point

    Source: http://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
    """

    image_size = (image.shape[1], image.shape[0])
    image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

    if (width > image_size[0]):
        width = image_size[0]

    if (height > image_size[1]):
        height = image_size[1]

    x1 = int(image_center[0] - width * 0.5)
    x2 = int(image_center[0] + width * 0.5)
    y1 = int(image_center[1] - height * 0.5)
    y2 = int(image_center[1] + height * 0.5)

    return image[y1:y2, x1:x2]


class RotNetLoader(Dataset):
    def __init__(self, dirName, imagesize, onehot=False):
        super(RotNetLoader, self).__init__()
        self.dirName = dirName
        self.imageSize = imagesize
        self.onehot = onehot
        self.imageList = os.listdir(self.dirName)
        self.imageList = [os.path.join(self.dirName, file) for file in self.imageList]

    def __len__(self):
        return len(self.imageList)

    def imageRote(self, imgfile):
        rotAngle = np.random.randint(360)

        # rotAngle = 0
        image = cv2.imread(imgfile)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rotImg = RotateImage(image, rotAngle)
        h, w = image.shape[:2]
        rotImg = CenterCrop(rotImg, *largest_rotated_rect(w, h, rotAngle))
        rotImg = cv2.resize(rotImg, (self.imageSize, self.imageSize))
        # self.showImg(image, rotImg, rotAngle)
        return rotImg, rotAngle

    def showImg(self, img, rotimage, angle):
        rotImg2 = RotateImage(rotimage, -angle)
        plt.subplot(221)
        plt.imshow(img)
        plt.subplot(222)
        plt.imshow(rotimage)
        plt.title(angle)
        plt.subplot(223)
        plt.imshow(rotImg2)
        plt.title(-angle)
        plt.show()

    def __getitem__(self, item):
        imgFile = self.imageList[item]
        rotImg, rotAngle = self.imageRote(imgFile)
        rotImg = (rotImg / 255.0).transpose(2, 0, 1)
        if self.onehot:
            rotAngle = np.array(rotAngle)
        else:
            rotAngle = np.array(rotAngle / 360)
        return rotImg, rotAngle


def collate_fn(batch):
    IMAGE = []
    LABEL = []
    for item in batch:
        img, label = item
        IMAGE.append(img)
        # LABEL.append(label.reshape(1, 1))
        LABEL.append(label)
    IMAGE = np.array(IMAGE)
    # LABEL = np.concatenate(LABEL, 0)
    LABEL = np.array(LABEL)
    return torch.from_numpy(IMAGE), torch.from_numpy(LABEL)


if __name__ == '__main__':
    loder = RotNetLoader(dirName=r"\part2", imagesize=512)
    for _ in range(10):
        batch = next(iter(loder))
        img, angle = batch
        print(img.shape, angle)
