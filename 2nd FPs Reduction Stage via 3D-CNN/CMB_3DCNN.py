
"""
Created on Friday October 11 2019

@author: Mohammed Al-masni
"""
# The CMB 3D-CNN
###############################
import glob
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import np_utils, to_categorical,plot_model
from keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau

from nilearn import image

from sklearn import preprocessing
from keras.models import Model, model_from_json

from keras.optimizers import SGD, Adam, Adadelta


from keras import backend as K

from keras.preprocessing.image import ImageDataGenerator
from keras.backend import categorical_crossentropy

# For 3D CNN -------------------------------------------------------
from keras.layers.convolutional import Conv3D, ZeroPadding3D
from keras.layers.convolutional import MaxPooling3D, AveragePooling3D

from keras.layers import Dropout, Input
from keras.layers import Flatten, add, Concatenate, GlobalAveragePooling3D, Lambda
from keras.layers import Dense

from keras.layers import BatchNormalization

from keras.layers import Activation 
from keras import regularizers
from keras import backend
# -------------------------------------------------------


#------------------------------------------------------------------------------
Train = 1 # True False    
Test  = 1 # True False

epoch = 100
learningRate = 0.001 # 0.001
decay = learningRate/epoch
optimizer = Adam(lr=learningRate)
batch_size = 50

Height = 16#32
Width  = 32#32
Depth  = 16#32
shape  = [1, Height, Width, Depth]

def read_data(class_names,class_labels):
	fold1_test  = list()  
	ts_lbl      = list()  
	fold2_train = list()  
	tr_lbl      = list() 
	fold3_valid = list()  
	val_lbl     = list()  

	for pos,sel in enumerate(class_names):
		print(pos,sel)
		images_test  = sorted(glob.glob("../data/Testing/"+sel+"/*.nii"))  #Testing 
		images_train = sorted(glob.glob("../data/Training/"+sel+"/*.nii")) #Training
		images_valid = sorted(glob.glob("../data/Testing/"+sel+"/*.nii"))  #Validation

	#Testing database:-------------------------------------------------------------
		for volume in images_test:
			img = image.load_img(volume)   
			img = img.get_data()    # convert to array
			fold1_test.append(np.asarray(img, dtype = np.float32) / 1.)   # Input Data is already normalized between 0~1
			ts_lbl.append(class_labels[pos])

	#Training database: -----------------------------------------------------------
		for volume in images_train:
			img = image.load_img(volume)
			img = img.get_data()    # convert to array
			fold2_train.append(np.asarray(img, dtype = np.float32) / 1.)  # Input Data is already normalized between 0~1
			tr_lbl.append(class_labels[pos])
	

	#Validation database: -----------------------------------------------------------
		for volume in images_valid:
			img = image.load_img(volume)
			img = img.get_data()    # convert to array
			fold3_valid.append(np.asarray(img, dtype = np.float32) / 1.)  # Input Data is already normalized between 0~1
			val_lbl.append(class_labels[pos])
    
	x_train = np.asarray(fold2_train)   
	y_test  = np.asarray(fold1_test) 
	y_valid = np.asarray(fold3_valid) 

	tr_lbl  = np.asarray(tr_lbl)
	ts_lbl  = np.asarray(ts_lbl)
	val_lbl = np.asarray(val_lbl)
   
	# in case of 'channels_first'
	x_train = x_train.reshape(x_train.shape[0], 1, Height, Width, Depth)
	y_test  = y_test.reshape(y_test.shape[0], 1, Height, Width, Depth)
	y_valid = y_valid.reshape(y_valid.shape[0], 1, Height, Width, Depth)
	
	tr_onehot  = to_categorical(tr_lbl)  # Converts a class vector (integers) to binary class matrix representation
	ts_onehot  = to_categorical(ts_lbl)
	val_onehot = to_categorical(val_lbl)
	
	return x_train, y_test, y_valid, tr_onehot, ts_onehot, val_onehot
	

#-- This part to save and load the MODEL weights:------------------------------
def save_model(model,md = 'lstm'):
	model_json = model.to_json()
	with open("model_"+md+".json", "w") as json_file:
		json_file.write(model_json)
	model.save_weights("model_"+md+".h5")
	print("The model is successfully saved")

def load_model(md = 'lstm'):
	json_file = open("model_"+md+".json", 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	loaded_model.load_weights("model_"+md+".h5")
	print("Loaded model from disk")
	return loaded_model

def Create_3DCNN(shape, classes):
	inpt = Input(shape=shape)
	x = ZeroPadding3D((1, 1, 1), data_format='channels_first')(inpt) 

	# conv1
	x = Conv3D(nb_filter=32, kernel_size=(3, 3, 3), strides=1, padding='valid',
				data_format='channels_first')(x)
	x = Activation('relu')(x)
	x = BatchNormalization()(x)	
	# conv2
	x = Conv3D(nb_filter=64, kernel_size=(3, 3, 3), strides=1, padding='same',
				data_format='channels_first')(x) 
	x = Activation('relu')(x)
	x = BatchNormalization()(x)
	# conv3
	x = Conv3D(nb_filter=64, kernel_size=(3, 3, 3), strides=1, padding='same',
				data_format='channels_first')(x) 
	x = Activation('relu')(x)
	x = BatchNormalization()(x)
	x = MaxPooling3D(pool_size=(2, 2, 2), strides=2, data_format='channels_first')(x)
	
	# conv4
	x = Conv3D(nb_filter=128, kernel_size=(3, 3, 3), strides=1, padding='same', 
				data_format='channels_first')(x) 
	x = Activation('relu')(x)
	x = BatchNormalization()(x)
	# conv5
	x = Conv3D(nb_filter=128, kernel_size=(3, 3, 3), strides=1, padding='same', 
				data_format='channels_first')(x) 
	x = Activation('relu')(x)
	x = BatchNormalization()(x)
	
	x = MaxPooling3D(pool_size=(2, 2, 2), strides=2, data_format='channels_first')(x)
	
	x = Flatten()(x)          
	x = Dense(32, activation='relu')(x)    
	x = Dropout(0.3)(x)    
	x = Dense(8, activation='relu')(x)     
	x = Dense(classes, activation='softmax')(x) # 
	model = Model(inputs=inpt, outputs=x)
   
	return model

def CMB_3DCNN_main():
	class_names = ['Non-CMB', 'CMB']
	class_labels = [0,1]
	train_data,test_data,valid_data, train_lbl,test_lbl, valid_lbl = read_data(class_names,class_labels)

	print('---------------------------------')
	print('Trainingdata=',train_data.shape)
	print('Validationdata=',valid_data.shape)
	print('Testingdata=',test_data.shape)
	print('Traininglabel=',train_lbl.shape)
	print('Validationlabel=',valid_lbl.shape)
	print('Testinglabel=',test_lbl.shape)
	print('---------------------------------')

	if Train:
		model = Create_3DCNN(shape, 2) 
		model.summary()
		model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])  

		print('Training Model')
		csv_logger = CSVLogger('Loss_Acc.csv', append=True, separator=' ')

		checkpoint = ModelCheckpoint('model-{epoch:03d}.h5', verbose=1, monitor='val_loss',save_best_only=True, mode='auto')
		
		# class weight for imbalance data:
		#Fold 1:
		class_weights = {0: 1.0, 
		                 1: 5.0}
		#Fold 2:
		#class_weights = {0: 1., 
		#                1: 12.64}
		#Fold 3:
		#class_weights = {0: 1., 
		#               1: 12.793}
		#Fold 4:
		#class_weights = {0: 1., 
		#                 1: 12.789}
		#Fold 5:
		#class_weights = {0: 1., 
		#                 1: 13.493}


		model.fit(x = train_data,
			y = train_lbl,
			batch_size = batch_size,
			shuffle=True,
			epochs = epoch, #100,
			verbose = 1,          # Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch 
			validation_data=(valid_data,valid_lbl),
			callbacks=[csv_logger, checkpoint],
			class_weight = class_weights)

		save_model(model,'CMB_3DCNN_') # to save the WEIGht 

	if Test:
	## Load the model and make file with predicted labels ##
		new_model = load_model('CMB_3DCNN_') # to load the weight 
		new_model.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics=['accuracy']) 
		pred = new_model.predict(test_data)
	
		## Evaluate the result and show an example ##
		print("Evaluate Model Normal ...")
		score = new_model.evaluate(test_data,test_lbl, verbose=1)
		print('Test loss:', score[0])
		print('Test accuracy:', score[1])
		#plt.figure()
		#plt.imshow(test_data[1,:,:,0])
		#plt.gray()
		#plt.show()
		print('Label original {} and predicted {}'.format(test_lbl[1],pred[1]))
		
		# Save predicted labels:-------------------------------  
		print('Data predicted for all model')
		Actual_label_test    = np.argmax(test_lbl,axis=1) # to make all class labels in one colum           # ??? argmax provide position of maximum value   ???
		Predicted_label_test = np.argmax(np.round(pred),axis=1)
		np.savetxt("../Results/Actual.csv", Actual_label_test, fmt='%.6f', delimiter=',') 
		np.savetxt("../Results/Predicted.csv", Predicted_label_test, fmt='%.6f', delimiter=',')

if __name__ == "__main__":
	CMB_3DCNN_main()  