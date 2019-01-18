import numpy as np
import pandas as pd
import os
import time

from keras.models import load_model
from sklearn.model_selection import train_test_split

import model_defs
from log_config import logger
import preprocessing

def train(X, Y, iteration):
	XTrain, XTest, yTrain, yTest = train_test_split(X, Y, test_size=0.33, random_state=42)

	if(os.path.exists('saved_models/agrix_alexnet.h5')):
		logger.info('Reading model from file. Time to improve!!')
		alexnet = load_model('saved_models/agrix_alexnet.h5')
	else:
		logger.info('Instantiating model for the first time')
		alexnet = model_defs.alexnet()


	alexnet.fit(XTrain, yTrain, batch_size = 64, epochs = 100, validation_split = 0.33, shuffle = True, verbose = 1)
	alexnet.save('saved_models/agrix_alexnet.h5')
	logger.info('Saved model to file.')

	scores = alexnet.evaluate(XTest, yTest, batch_size = 64, verbose = 1)
	logger.info("PERFORMANCE SCORE: {}: {}".format(alexnet.metrics_names[1], scores[1] * 100))





if __name__ == '__main__':
	# Training in batches because of the dataset size
	p = 0 
	q = 11
	iteration = 0
	top = 125 # The total pics in the class with fewest images + 1

	while (p < 38):
		logger.info('Reading DF...')
		df = preprocessing.main('train', p, q) #Let's do (q - p) pics from all classes per iter

		X = np.array(df['image'].tolist())
		df['vector_labels'] = pd.get_dummies(df['label']).values.tolist()
		Y = np.array(df['vector_labels'].tolist())

		logger.info('X.shape: {}'.format(X.shape))
		logger.info('Y.shape: {}'.format(Y.shape))
		logger.info('Starting training...')

		train(X,  Y, iteration)

		p = p + 10
		q = (q - 1) + 10 

		time.sleep(60)