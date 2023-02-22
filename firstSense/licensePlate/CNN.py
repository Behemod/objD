# -*- coding:utf-8 -*-
# original author: DuanshengLiu
from keras import layers, losses, models
import numpy as np
import cv2
import os


def cnn_train():
    char_dict = {"京": 0, "沪": 1, "津": 2, "渝": 3, "冀": 4, "晋": 5, "蒙": 6, "辽": 7, "吉": 8, "黑": 9, "苏": 10,
                 "浙": 11, "皖": 12, "闽": 13, "赣": 14, "鲁": 15, "豫": 16, "鄂": 17, "湘": 18, "粤": 19, "桂": 20,
                 "琼": 21, "川": 22, "贵": 23, "云": 24, "藏": 25, "陕": 26, "甘": 27, "青": 28, "宁": 29, "新": 30,
                 "0": 31, "1": 32, "2": 33, "3": 34, "4": 35, "5": 36, "6": 37, "7": 38, "8": 39, "9": 40,
                 "A": 41, "B": 42, "C": 43, "D": 44, "E": 45, "F": 46, "G": 47, "H": 48, "J": 49, "K": 50,
                 "L": 51, "M": 52, "N": 53, "P": 54, "Q": 55, "R": 56, "S": 57, "T": 58, "U": 59, "V": 60,
                 "W": 61, "X": 62, "Y": 63, "Z": 64}

    # read dataset
    path = 'home/cnn_datasets/'  # License plate number data set path (license plate picture width 240, height 80)
    pic_name = sorted(os.listdir(path))
    n = len(pic_name)
    X_train, y_train = [], []
    for i in range(n):
        print("Reading image %d" % i)
        img = cv2.imdecode(np.fromfile(path + pic_name[i], dtype=np.uint8), -1)  # cv2.imshow cannot read Chinese path pictures, use this method instead
        label = [char_dict[name] for name in pic_name[i][0:7]]  # The first 7 digits of the picture name are the license plate label
        X_train.append(img)
        y_train.append(label)
    X_train = np.array(X_train)
    y_train = [np.array(y_train)[:, i] for i in range(7)]  # y_train is a list of length 7, each of which is an ndarray of shape (n,), corresponding to the first character, second character....seventh character of n pictures

    # cnn model
    Input = layers.Input((80, 240, 3))  # license plate image shape(80,240,3)
    x = Input
    x = layers.Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(x)
    x = layers.MaxPool2D(pool_size=(2, 2), padding='same', strides=2)(x)
    for i in range(3):
        x = layers.Conv2D(filters=32 * 2 ** i, kernel_size=(3, 3), padding='valid', activation='relu')(x)
        x = layers.Conv2D(filters=32 * 2 ** i, kernel_size=(3, 3), padding='valid', activation='relu')(x)
        x = layers.MaxPool2D(pool_size=(2, 2), padding='same', strides=2)(x)
        x = layers.Dropout(0.5)(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.3)(x)
    Output = [layers.Dense(65, activation='softmax', name='c%d' % (i + 1))(x) for i in range(7)]  # The 7 outputs correspond to the 7 characters of the license plate, and each output is 65 category probabilities
    model = models.Model(inputs=Input, outputs=Output)
    model.summary()
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',  # y_train is not one-hot encoded, so loss chooses sparse_categorical_crossentropy
                  metrics=['accuracy'])

    # model training
    print("start training cnn")
    model.fit(X_train, y_train, epochs=35)  # The total loss is the sum of 7 losses
    model.save('cnn.h5')
    print('cnn.h5 saved successfully!!!')


def cnn_predict(cnn, Lic_img):
    characters = ["京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "皖", "闽", "赣", "鲁", "豫",
                  "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "0", "1", "2",
                  "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M",
                  "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
    Lic_pred = []
    for lic in Lic_img:
        lic_pred = cnn.predict(lic.reshape(1, 80, 240, 3))  # The predicted shape should be (1,80,240,3)
        lic_pred = np.array(lic_pred).reshape(7, 65)  # The list is converted to ndarray, the shape is (7,65)
        if len(lic_pred[lic_pred >= 0.8]) >= 4:  # Count the number of predicted probability values greater than 80%, and more than or equal to 4 are considered to have a high recognition rate and successful recognition
            chars = ''
            for arg in np.argmax(lic_pred, axis=1):  # Take the arg with the largest probability value in each line and convert it to a character
                chars += characters[arg]
            chars = chars[0:2] + '·' + chars[2:]
            Lic_pred.append((lic, chars))  # Store the license plate and recognition results together in Lic_pred
    return Lic_pred
