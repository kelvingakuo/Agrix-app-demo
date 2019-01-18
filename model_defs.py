from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Activation, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization


def alexnet():
	model = Sequential()
	#convolution layer 1
	model.add(Conv2D(96,kernel_size=(11,11),strides=(4,4),padding='valid', data_format = 'channels_last', input_shape = (227, 227, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
	model.add(BatchNormalization())

	#Convolution layer 2
	model.add(Conv2D(filters=256,kernel_size=(11,11),strides=(1,1),padding='valid'))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
	model.add(BatchNormalization())

	#Convolution layer 3
	model.add(Conv2D(filters=384,kernel_size=(3,3), strides=(1,1), padding='valid'))
	model.add(Activation('relu'))
	model.add(BatchNormalization())

	#Convolution layer 4
	model.add(Conv2D(filters=384,kernel_size=(3,3), strides=(1,1), padding='valid'))
	model.add(Activation('relu'))
	model.add(BatchNormalization())

	#convolution layer 5
	model.add(Conv2D(filters=256,kernel_size=(3,3), strides=(1,1), padding='valid'))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
	model.add(BatchNormalization())

	model.add(Flatten())

	#Fully connected layer 1
	model.add(Dense(4096, input_shape = (227, 227, 3), activation='relu'))
	model.add(Dropout(0.4))
	model.add(BatchNormalization())

	#Fully  connected layer 2
	model.add(Dense(4096, activation='relu'))
	model.add(Dropout(0.4))
	model.add(BatchNormalization())

	#Fully connected layer 3
	model.add(Dense(38,activation='relu')) # Output size of 38 classes
	model.add(Dropout(0.4))

	#output layer
	model.add(Activation('softmax'))

	model.compile(loss = 'categorical_crossentropy' ,optimizer='adam',metrics=['accuracy']) 

	return model


def Vgg16(inputs):
	X = Input(inputs)

	#Convolution layer 1
	conv_1 = Conv2D(64,(3,3), padding='same',)(X)
	conv_1 = Activation('relu')(conv_1)
	conv_1 = Conv2D(64,(3,3),padding='same')(conv_1)
	conv_1 = Activation('relu')(conv_1)
	conv_1 = MaxPooling2D(pool_size=(2,2),strides=(2,2))(conv_1)	

	#convolution layer 2
	conv_2 = Conv2D(128,(3,3), padding='same')(conv_1)
	conv_2 = Activation('relu')(conv_2)
	conv_2 = Conv2D(128,(3,3),padding='same')(conv_2)
	conv_2 = Activation('relu')(conv_2)
	conv_2 = MaxPooling2D(pool_size=(2,2),strides=(2,2))(conv_2)

	#convolution layer 3
	conv_3 = Conv2D(256,(3,3), padding='same')(conv_2)
	conv_3 = Activation('relu')(conv_3)
	conv_3 = Conv2D(256, (3,3), padding='same')(conv_3)
	conv_3 = Activation('relu')(conv_3)
	conv_3 = MaxPooling2D(pool_size=(2,2),strides=(2,2))(conv_3)

	#convolution layer 4
	conv_4 = Conv2D(512,(3,3), padding='same')(conv_3)
	conv_4 = Activation('relu')(conv_4)
	conv_4 = Conv2D(512,(3,3),padding='same')(conv_4)
	conv_4 = Activation('relu')(conv_4)
	conv_4 = Conv2D(512,(3,3),padding='same')(conv_4)
	conv_4 = Activation('relu')(conv_4)
	conv_4 = MaxPooling2D(pool_size=(2,2),strides=(2,2))(conv_4)

	#convolution layer 5
	conv_5 = Conv2D(512,(3,3),padding='same')(conv_4)
	conv_5 = Activation('relu')(conv_5)
	conv_5 = Conv2D(512,(3,3),padding='same')(conv_5)
	conv_5 = Activation('relu')(conv_5)
	conv_5 = Conv2D(512,(3,3),padding='same')(conv_5)
	conv_5 = Activation('relu')(conv_5)
	conv_5 = MaxPooling2D(pool_size=(2,2),strides=(2,2))(conv_5)

	flat = Flatten()(conv_5)

	#fully connected layer 1
	dense_1 = Dense(4096)(flat)
	dense_1 = Activation('relu')

	#fully connected layer 2
	dense_2 = Dense(4096)(dense_1)
	dense_2 = Activation('relu')(dense_2)

	#output
	output = Dense(1000)(dense_2)
	output = Activation('softmax')(output)

	model = Model(input = inputs, output=output)

	model.compile(loss=categorical_crossentropy, optimizer='adam', metrics=['accuracy'])  


