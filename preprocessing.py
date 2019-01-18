import cv2
import glob
import numpy as np
import pandas as pd
from pathlib import Path
import os
import re
import sys


from log_config import logger


def main(setType, nn, a = None, b = None):
	if(setType == 'train'):
		folders = glob.glob('data/crowdai_train/crowdai/*') # Get folder names
		parts = [Path(folder).parts[3] for folder in folders]
		classes = [re.search(r'c_(.*)', part, re.I|re.M).group(1) for part in parts] #Get the classes


		train_data = dict()
		train_data['image'] = []
		train_data['label'] = []
		train_data['location'] = []

		logger.info('Creating train arrays for {} ...'.format(nn))

		if(nn == 'alexnet'):
			dim = 227
		elif(nn == 'vgg16'):
			dim = 224

		j = 0 # Classes index
		while(j < len(classes)):
			lbl = classes[j]
			logger.info('Opening folder: {} ...'.format(lbl))
			folder = folders[j]

			if((a is None) & (b is None)):
				imgs = glob.glob(folder+'/*.jpg')
			else:
				imgs = glob.glob(folder+'/*.jpg')[a:b] #b - a pics

			for img in imgs:
				logger.info('Creating data point {} ...'.format(img))

				rd = cv2.imread(img)
				arr = cv2.resize(rd, (dim, dim))

				arr = arr.astype('uint64')

				train_data['image'].append(arr)
				train_data['label'].append(lbl)
				train_data['location'].append(folder)

				logger.info('Finished creating data point {}.'.format(img))

			j = j + 1


		logger.info('Finished creating arrays. Writing to DF...')
		np.set_printoptions(threshold = np.inf)
		df = pd.DataFrame.from_dict(train_data, orient = 'columns')
		logger.info('Finished writing to DF.')

		return df


# Deal with test data
	elif(setType == 'test'):

		test_data = dict()
		test_data['image'] = []

		imgs = glob.glob('data/crowdai_test/test/*.jpg')

		logger.info('Creating test arrays...')

		for img in imgs:
			logger.info('Creating data point {} ...'.format(img))
			
			rd = cv2.imread(img)
			arr = cv2.resize(rd, (227, 227))

			test_data['image'].append(arr)
			logger.info('Finished creating data point {} ...'.format(img))


		logger.info('Finished creating arrays. Writing to DF...')
		np.set_printoptions(threshold = np.inf)
		df = pd.DataFrame.from_dict(train_data, orient = 'columns')
		logger.info('Finished writing to DF.')

		return df





