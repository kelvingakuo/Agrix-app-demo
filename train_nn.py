import numpy as np
import pandas as pd
import os
import sys
import time


from keras.models import load_model
from sklearn.model_selection import train_test_split

import model_defs
from log_config import logger
import preprocessing

def train(X, Y, iteration, whichOne):
	XTrain, XTest, yTrain, yTest = train_test_split(X, Y, test_size=0.33, random_state=42)

	if(whichOne == 'alexnet'):
		if(os.path.exists('saved_models/agrix_alexnet.h5')):
			logger.info('Reading alexnet from file. Time to improve!!')
			model = load_model('saved_models/agrix_alexnet.h5')
		else:
			logger.info('Instantiating model for the first time')
			model = model_defs.alexnet()
			
	elif(whichOne == 'vgg16'):
		if(os.path.exists('saved_models/agrix_vgg16.h5')):
			logger.info('Reading vgg16 from file. Time to improve!!')
			model = load_model('saved_models/agrix_vgg16.h5')
		else:
			logger.info('Instantiating model for the first time')
			model = model_defs.vgg16()


	# Run model
	model.fit(XTrain, yTrain, batch_size = 64, epochs = 100, validation_split = 0.33, shuffle = True, verbose = 1)
	scores = model.evaluate(XTest, yTest, batch_size = 64, verbose = 1)
	logger.info("VALIDATION SCORE: {}: {}%".format(model.metrics_names[1], scores[1] * 100))

	if(whichOne == 'alexnet'):
		model.save('saved_models/agrix_alexnet.h5')
		logger.info('Saved alexnet to file.')
	elif(whichOne == 'vgg16'):
		model.save('saved_models/agrix_vgg16.h5')
		logger.info('Saved vgg16 to file.')







if __name__ == '__main__':
	"""
	Due to RAM constraints, I'm reading the images in batches of 10 i.e. 10 images per class per iteration.

	If access to GPU or a massive RAM, you can: 
		1. Set p = None and q = None
		2. Set p = 0 and q = (n + 1) where the n is the number of pictures to read per iteration
			Make sure to update p and q by replacing 10 with n
	"""
	# Training in batches because of the dataset size
	p = 0 
	q = 21
	iteration = 0
	top = 125 # The total pics in the class with fewest images + 1

	theModel = sys.argv[1]

	while (p < top):
		logger.info('Reading DF...')
		df = preprocessing.main('train', theModel, p, q) #Let's do (q - p) pics from all classes per iter

		X = np.array(df['image'].tolist()) # Generate array of arrays for X, and array of vectors for y
		df['vector_labels'] = pd.get_dummies(df['label']).values.tolist()
		Y = np.array(df['vector_labels'].tolist())

		logger.info('X.shape: {}'.format(X.shape))
		logger.info('Y.shape: {}'.format(Y.shape))
		logger.info('Starting training...')

		train(X,  Y, iteration, theModel)

		if(p is None):
			break
		else:
			p = p + 20
			q = q + 20 
			iteration = iteration + 1

			time.sleep(60)