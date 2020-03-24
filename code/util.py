import numpy as np
from glob import glob
import cv2
import os
import random 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split 

IMG_LIST = []
LABEL_LIST = []

def normalize(x):
    min_val = np.min(x)
    max_val = np.max(x)
    x = (x-min_val) / (max_val-min_val)
    return x

def next_batch(num, data, labels):
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]
    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

def one_hot_encode(x, NROFCLASSES):
    encoded = np.zeros((len(x), NROFCLASSES))
    for idx, val in enumerate(x):
        encoded[idx][val] = 1
    return encoded

def read_train_valid_data(directory):
    for img_file in glob(directory + "/s0/*.png"):
        nsfw_label = 0
        img = cv2.imread(img_file)
        try:
            img = cv2.resize(img, (224, 224))
            IMG_LIST.append(img)
            LABEL_LIST.append(nsfw_label)
        except:
            print('Error at {}'.format(img_file))
    for img_file in glob(directory + "/s1/*.png"):
        img = cv2.imread(img_file)
        sfw_label = 1
        try:
            img = cv2.resize(img, (224, 224))
            IMG_LIST.append(img)
            LABEL_LIST.append(sfw_label)
        except:
            print('Error at {}'.format(img_file))
    np.save('images_224.npy', IMG_LIST)
    np.save('labels_224.npy', LABEL_LIST)
    assert(len(IMG_LIST) == len(LABEL_LIST))

def train_test_split_data(NROFCLASSES):
    mapping_file = open("mapping_file_labels.txt", "w+")
    IMAGES = np.load('/content/images_224.npy')
    LABELS = np.load('/content/labels_224.npy')
    one_hot_labels = one_hot_encode(LABELS, NROFCLASSES)
    x_train, x_val, y_train, y_val = train_test_split(IMAGES, one_hot_labels, test_size=0.2, random_state=42)
    x_train = np.asarray(x_train)
    x_val = np.asarray(x_val)
    y_val = np.asarray(y_val)
    y_train = np.asarray(y_train)
    print("Done Data Formation.")
    print('X_TRAIN, Y_TRAIN Shape', x_train.shape, y_train.shape)
    print('X_VAL, Y_VAL Shape', x_val.shape, y_val.shape)
    return x_train, x_val, y_train, y_val
