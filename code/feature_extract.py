


#! /usr/bin/python
# -*- coding: utf-8 -*-


import numpy as np
import cPickle 
from collections import OrderedDict
import csv
import glob
import os
from numpy import mean, sqrt, square
import scipy.stats as sts
from subprocess import Popen, PIPE
from pywt import WaveletPacket



def rms_fea(a):
	return np.sqrt(np.mean(np.square(a)))

def var_fea(a):
	return np.var(a)

def max_fea(a):
	return np.max(a)

def pp_fea(a):
	return np.max(a)-np.min(a)

def skew_fea(a):
	return sts.skew(a)

def kurt_fea(a):
	return sts.kurtosis(a)

def wave_fea(a):
	wp = WaveletPacket(a,'db1', maxlevel=8)
	nodes = wp.get_level(8, "freq")
	return np.linalg.norm(np.array([n.data for n in nodes]), 2)

def spectral_kurt(a):
	N= a.shape[0]
	mag = np.abs(np.fft.fft(a))
	mag	= mag[1:N/2]*2.00/N
	return sts.kurtosis(mag)

def spectral_skw(a):
	N= a.shape[0]
	mag = np.abs(np.fft.fft(a))
	mag	= mag[1:N/2]*2.00/N
	return sts.skew(mag)

def spectral_pow(a):
	N= a.shape[0]
	mag = np.abs(np.fft.fft(a))
	mag	= mag[1:N/2]*2.00/N
	return np.mean(np.power(mag, 3))


def extract_fea(data, num_stat =5):
	# input: time_len * dim_fea  -> dim_fea*9
	data_fea = []
	dim_feature = 1
	for i in range(dim_feature):
		data_slice = data
		data_fea.append(rms_fea(data_slice))
		data_fea.append(var_fea(data_slice))
		#data_fea.append(max_fea(data_slice))
		#data_fea.append(pp_fea(data_slice))
		#data_fea.append(skew_fea(data_slice))
		data_fea.append(kurt_fea(data_slice))
		#data_fea.append(wave_fea(data_slice))
		data_fea.append(spectral_skw(data_slice))
		data_fea.append(spectral_kurt(data_slice))
		#data_fea.append(spectral_pow(data_slice))		
	data_fea = np.array(data_fea)
	return data_fea.reshape((1,dim_feature*num_stat))
		
def gen_fea(data,time_steps=20,num_stat = 5):
	data_num = data.shape[0]
	len_seq = data.shape[1]
	window_len = len_seq/time_steps
	if window_len == len_seq:
		new_data = np.ones((data_num,num_stat), dtype=np.float32)
		for idxdata, sig_data in enumerate(data):
			new_data[idxdata,:] = extract_fea(sig_data)
	else:
		new_data = np.ones((data_num,time_steps,num_stat), dtype=np.float32)
		for idxdata, sig_data in enumerate(data):
			for i in range(time_steps):
				start = i*window_len
				end = (i+1)*window_len		
				temp_data = extract_fea(sig_data[start:end])
				new_data[idxdata,i,:] = temp_data
	return new_data



def short_data(data_x, data_y):
	new_len = 1000
	data_new_x = np.zeros((data_x.shape[0]*4, new_len))
	data_new_y = np.zeros((data_x.shape[0]*4,))
	for ith in range(data_x.shape[0]):
		data_new_x[ith*4] =  data_x[ith,0:new_len]
		data_new_y[ith*4] = data_y[ith]  
		data_new_x[ith*4+1] = data_x[ith,new_len:2*new_len]
		data_new_y[ith*4+1] = data_y[ith]  
		data_new_x[ith*4+2] = data_x[ith,2*new_len:3*new_len]
		data_new_y[ith*4+2] = data_y[ith] 
		data_new_x[ith*4+3] = data_x[ith,3*new_len:4*new_len]
		data_new_y[ith*4+3] = data_y[ith]  
	return data_new_x, data_new_y


def main():
	
	x = cPickle.load(open("../data/cwru.p","rb"))
	data_x, data_y = x[0], x[1]
	print 'data loaded'
	new_data_x, new_data_y = short_data(data_x, data_y)
	data_time = gen_fea(new_data_x)
	data_normal = gen_fea(new_data_x, 1)
	len_data = data_normal.shape[0]
	cv_idx = np.random.randint(3, size=len_data)
	cPickle.dump([new_data_x, data_time, data_normal, new_data_y, cv_idx], open('../data/cwrupuredata.p', "wb"))

if __name__ == '__main__':
	main()