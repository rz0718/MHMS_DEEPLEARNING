"""
This script define various feature extraction methods
"""
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


def extract_fea(data, num_stat = 10):
	# input: time_len * dim_fea  -> dim_fea*9
	data_fea = []
	dim_feature = 1
	for i in range(dim_feature):
		data_slice = data
		data_fea.append(rms_fea(data_slice))
		data_fea.append(var_fea(data_slice))
		data_fea.append(max_fea(data_slice))
		data_fea.append(pp_fea(data_slice))
		data_fea.append(skew_fea(data_slice))
		data_fea.append(kurt_fea(data_slice))
		data_fea.append(wave_fea(data_slice))
		data_fea.append(spectral_kurt(data_slice))
		data_fea.append(spectral_skw(data_slice))
		data_fea.append(spectral_pow(data_slice))
	data_fea = np.array(data_fea)
	return data_fea.reshape((1,dim_feature*num_stat))
		
def gen_fea(data,time_steps = 20,num_stat = 10):
	"""
	input: 
		@data: raw time series data, [data size,  sequence length]
		@time_steps: the number of windows, and it can be one
		@num_stat: number of features for each window
	"""
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
