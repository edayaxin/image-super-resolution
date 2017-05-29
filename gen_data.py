# -*- coding: utf-8 -*-
import h5py
import numpy as np
import glob 
import cv2


def modcrop(img, modulo):
    sz = np.shape(img)
    sz = sz - np.mod(sz, modulo)
    return img[0 : sz[0] - 1, 0 : sz[1] - 1]


def store2hdf5(filename, data, labels):

    f = h5py.File(filename, "w")

    f.create_dataset('data', data=data)
    f.create_dataset('label', data=labels)
    f.close()    


folder = './Train/'
savepath = 'examples/SRCNN/train.h5'
size_input = 33
size_label = 21
scale = 3
stride = 14

max_sample = 30000

data = np.reshape(np.array([]), (size_input, size_input, 1, 0))
label = np.reshape(np.array([]), (size_label, size_label, 1, 0))

data = np.zeros((size_input, size_input, 1, max_sample))
label = np.zeros((size_label, size_label, 1, max_sample))

padding = abs(size_input - size_label) / 2


filepaths = glob.glob(folder + '*.bmp')

count = 0
for f in filepaths:
    image = cv2.imread(f)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB);
    image = image[:, :, 0] / 255.0

    im_label = modcrop(image, scale)
    sz = np.shape(im_label)
    #im_input = cv2.resize(cv2.resize(im_label, (sz[1]/scale, sz[0]/scale)), (sz[1], sz[0]))
    im_input = cv2.resize(cv2.resize(im_label, (sz[1]/scale, sz[0]/scale), interpolation=cv2.INTER_AREA), (sz[1], sz[0]), interpolation=cv2.INTER_CUBIC);     
    
    for x in range(0, stride, sz[0]-size_input):
        for y in range(0, stride, sz[1]-size_input):
            label[:,:,0,count] = im_label[x+padding:x+padding+size_label, y+padding:y+padding+size_label]
            data[:,:,0,count] = im_input[x:x+size_input, y:y+size_input]
            count = count + 1
    
data = data[:,:,:,0:count]
label = label[:,:,:,0:count]


"""
TODO #1:
Randomly permute the data pairs.
"""
order = np.random.permutation(count)
data = data[:,:,:,order]
label = label[:,:,:, order]

data = np.transpose(data, (3, 2, 1, 0))
label = np.transpose(label, (3, 2, 1, 0))

store2hdf5(savepath, data, label)