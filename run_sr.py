# -*- coding: utf-8 -*-

import numpy as np
import caffe
import cv2
import os
import glob 

# config paths
model_path = "examples/SRCNN/SRCNN_test.prototxt"
param_path = "examples/SRCNN/SRCNN_iter_1000000.caffemodel"

folder = './Test/Set14/'
result_folder = './Result/'
filepaths = glob.glob(folder + '*.bmp')

shrink = 12
count = 0
PSNR_sum = 0
PSNR_list = {}

PSNR_blur_sum = 0
PSNR_rec_sum = 0
PSNR_blur_list = {}
PSNR_rec_list = {}

for f in filepaths:
	net = caffe.Net(model_path, param_path, caffe.TEST)

	img = cv2.imread(f)
	sz = np.shape(img)

	#resize the image
	img = cv2.resize(cv2.resize(img, (sz[1]/3, sz[0]/3), interpolation=cv2.INTER_AREA), (sz[1], sz[0]), interpolation=cv2.INTER_CUBIC);     

	out = np.zeros((sz[0] - shrink, sz[1] - shrink, 3))
	inp = np.zeros((sz[0] - shrink, sz[1] - shrink, 3))
	# super resolution for each channel
	for i in range(0, 3):
	    ch = img[:,:,i] / 255.0;

	    net.blobs['data'].reshape(1, 1, sz[0], sz[1])
	    net.blobs['data'].data[...] = np.reshape(ch, (1, 1, sz[0], sz[1]));
	    net.forward()
	    x = net.blobs['conv3'].data[...]
	    out[:,:,i] = np.squeeze(x)

	# save outputs
	base =  os.path.basename(f)
	name = os.path.splitext(base)[0]
	output_name = result_folder + name + '_output.png'
	input_name = result_folder + name + '_input.png'
	print output_name, input_name
	inp = img[shrink / 2 : sz[0] - shrink / 2, shrink / 2 : sz[1] - shrink / 2, :]
	cv2.imwrite(input_name, img[shrink / 2 : sz[0] - shrink / 2, shrink / 2 : sz[1] - shrink / 2, :])
	cv2.imwrite(output_name, out * 255)
	w = inp.shape[0]
	h = inp.shape[1]

	inp = cv2.cvtColor(inp, cv2.COLOR_BGR2YCR_CB)
	print out.shape
	print out.dtype

	outimg = np.array(out, dtype=np.float32)
	outimg *= 255.0
	out = cv2.cvtColor(outimg, cv2.COLOR_BGR2YCR_CB)
	
	A = inp[:, :, 0] / 255.0;
	B = out[:, :, 0] / 255.0;
	mse = 0

	ori = cv2.imread(f)	
	ori = cv2.cvtColor(ori, cv2.COLOR_BGR2YCR_CB)
	C = ori[:, :, 0] / 255.0;
	ori_blur = 0
	ori_rec = 0

	for x in range(0, w):
		for y in range(0, h):
			sub = np.abs(A[x, y] - B[x, y])
			mse = mse + np.power(sub, 2)

			sub_blur = np.abs(C[x, y] - A[x, y])
			ori_blur = ori_blur + np.power(sub_blur,2)

			sub_rec = np.abs(C[x, y] - B[x, y])
			ori_rec = ori_rec + np.power(sub_rec, 2)

	mse = mse / (w*h)
	psnr = 10*np.log10(1/mse)
	PSNR_sum = PSNR_sum + psnr
	PSNR_list = np.append(PSNR_list, psnr)
	
	count = count + 1

	ori_blur = ori_blur / (w*h)
	ori_blur = 10*np.log10(1/ori_blur)
	PSNR_blur_list = np.append(PSNR_blur_list, ori_blur)
	PSNR_blur_sum = PSNR_blur_sum + ori_blur

	ori_rec = ori_rec / (w*h)
	ori_rec = 10*np.log10(1/ori_rec)
	PSNR_rec_list = np.append(PSNR_rec_list, ori_rec)
	PSNR_rec_sum = PSNR_rec_sum + ori_rec

print "blur_rec_average ", "ori_rec_average ", "ori_blur_average"
print PSNR_sum/count, " ", PSNR_rec_sum/count, " ", PSNR_blur_sum/count

print "blur_rec_each ", "ori_rec_each ", "ori_blur_each"
for i in range(1, len(PSNR_list)):
	print PSNR_list[i], " ", PSNR_rec_list[i], " ", PSNR_blur_list[i]


