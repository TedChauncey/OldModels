#KERAS
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam
from keras.utils import np_utils
from keras.models import load_model

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import theano
from PIL import Image


# SKLEARN
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from tempfile import TemporaryFile
import pandas as pd
from sklearn.metrics import roc_curve, auc

#from PIL import Image
from numpy import *
from keras import backend as K



#input image dimensions
OG_size = 150
img_rows, img_cols = 50, 50 # 50, 50 #normally. make sure this is an even number
Center = OG_size/2
x1, y1, x2, y2 = Center-img_rows/2,Center-img_rows/2,Center+img_cols/2,Center+img_cols/2 # 50, 50, 100, 100

# number of channels
img_channels = 1

# data

Outcomes_file = '/home/chintan/Desktop/AhmedData/EarlyStageSurvival.csv' # or StageISurvival.csv
path1 = '/home/chintan/Desktop/AhmedData/EarlyStageSurvival' # or StageISurvival    
path2 = '/home/chintan/Desktop/AhmedData/TestImageCrops'  #DELETE  

listing = os.listdir(path1) 
num_samples=size(listing)
print num_samples


for file in listing:
    im = Image.open(path1 + '/' + file) 
    img = im.crop((x1,y1,x2,y2))  #crop the images to x1-x2 x y1-y2 centered on tumor
    #img = im.resize((img_rows,img_cols))
    gray = img.convert('RGB')
                #need to do some more processing here           
    gray.save(path2 +'/' +  file, "PNG")

imlist = os.listdir(path2)
imlist.sort() #make sure to SORT the images to match 

im1 = array(Image.open( path2+ '/'+ imlist[0])) # open one image to get size
m,n = im1.shape[0:2] # get the size of the images
imnbr = len(imlist) # get the number of images

# create matrix to store all flattened images
immatrix = array([array(Image.open(path2+ '/' + im2)).flatten()
              for im2 in imlist],'f')             
              
#define labels as the outcomes

Outcomes = pd.read_csv(Outcomes_file) #create seperate outcomes file for TEST data
Outcome_of_interest = pd.Series.as_matrix(Outcomes.loc[:,'surv2yrTC']) #pick the column with the labels of interest
PID = pd.Series.as_matrix(Outcomes.loc[:,'patient'])

label = Outcome_of_interest

data,Label = immatrix,label
train_data = [data,Label]

print (train_data[0].shape)
print (train_data[1].shape)

#MODEL#

#batch_size to train
batch_size = 32
# number of output classes
nb_classes = 2
# number of epochs to train
nb_epoch = 30

from keras import backend as K #this just makes stuff work
K.set_image_dim_ordering('th')
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

(X, y) = (train_data[0],train_data[1])

X = X.reshape(X.shape[0], img_rows, img_cols,3)

X= X.astype('float32')

X /= 255

print('X shape:', X.shape)
print(X.shape[0], 'test samples')


# convert class vectors to binary class matrices
Y = np_utils.to_categorical(y, nb_classes)

#load pretrained model

predDir = '/home/chintan/Desktop/FinalModels'
#modelFile = (os.path.join(predDir,'VGG_SurvivalModel_RightCensored_run2_BESTMODEL_2_6_18.h5'))  #make sure the model is saved here. 
modelFile = (os.path.join(predDir,'VGG_Abstract_model2.h5'))  #VGG_Abstract_model2.h5 is the new killer model
model = load_model(modelFile)

#predictions using model

score = model.evaluate(X, Y, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


from sklearn.metrics import roc_curve, auc

def AUC(test_labels,test_prediction,nb):

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(nb):
        # ( actual labels, predicted probabilities )
        fpr[i], tpr[i], _ = roc_curve(test_labels[:, i], test_prediction[:, i] ) # flip here
        roc_auc[i] = auc(fpr[i], tpr[i])

    return [ round(roc_auc[x],3) for x in range(nb) ] 

Y_pred = model.predict(X)

ROC2 = AUC(Y, Y_pred, nb_classes)

print('AUC:',ROC2[1])

#Plot ROC curve
def AUCalt( test_labels , test_prediction):
    # convert to non-categorial
    test_prediction = np.array( [ x[1] for x in test_prediction   ])
    test_labels = np.array( [ 0 if x[0]==1 else 1 for x in test_labels   ])
    # get rates
    fpr, tpr, thresholds = roc_curve(test_labels, test_prediction, pos_label=1)
    # get auc
    myAuc = plt.plot(fpr, tpr)
    plt.xlabel('Specificity')
    plt.ylabel('Sensitivity')
    plt.title('ROC Curve')
    plt.grid(True)
    plt.legend(['VGG-16','SVM','RFC','MVR'],loc=4)
    plt.show()
    #myAuc = auc(fpr, tpr)
    return myAuc
   
plt.figure(1)    
ROC_VGG = AUCalt(Y, Y_pred)

plt.figure(2)
plt.hist(Y_pred[:,1])
plt.xlabel('Probability')
plt.ylabel('Incidence')
plt.title('Distribution of prediction probabilities Lung1 Stage 1')

## calculate sensitivity and specificity ##

from sklearn.metrics import confusion_matrix
from fractions import Fraction

cm1 = confusion_matrix(y, np.round(Y_pred[:,1]))
print('Confusion Matrix : \n', cm1)

total1=sum(sum(cm1))

accuracy1= Fraction((cm1[0,0]+cm1[1,1])/total1)
print ('Accuracy : ', accuracy1)

sensitivity1 = Fraction(cm1[0,0]/(cm1[0,0]+cm1[0,1]))
print('Sensitivity : ', sensitivity1 )

specificity1 = Fraction(cm1[1,1]/(cm1[1,0]+cm1[1,1]))
print('Specificity : ', specificity1)
         
