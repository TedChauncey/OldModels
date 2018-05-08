## This script will use a deep neural network as a fixed feature extractor ##
## ML classifiers are then used for histology classification ##
## Author: Tafadzwa Chaunzwa 2.9.18 ##

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
import os
import theano
from PIL import Image
from numpy import *

from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from tempfile import TemporaryFile
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif, SelectFromModel
from sklearn import linear_model
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD,RMSprop,adam
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from keras.models import Model

num_best_features = 60 #set to 60 to capture 90% of variance 29 for 75% variance, 10 for 50% of variance
test_size = .3630 

## Define fixed feature extractor ##
deep_model = VGG16(weights='imagenet', include_top=False, input_shape = (50,50,3) )

## Define the DATA! 

#input image dimensions
OG_size = 150 #original image size 
img_rows, img_cols = 50, 50 # 50, 50 #normally. make sure this is an even number
Center = OG_size/2
x1, y1, x2, y2 = Center-img_rows/2,Center-img_rows/2,Center+img_cols/2,Center+img_cols/2 # 50, 50, 100, 100

# number of channels
img_channels = 3

## DATA SOURCES ##

Outcomes_file = '/home/chintan/Desktop/MLPlayGround/Test_Set/Histology/AllOutcomes.csv' #define the outcomes file, sorted according to PID
path1 = '/home/chintan/Desktop/MLPlayGround/Test_Set/Histology/ALLImages'    #path of folder of images    
path2 = '/home/chintan/Desktop/MLPlayGround/Test_Set/Histology/ALLImageCrops' #path of folder to save images    

listing = os.listdir(path1) 
num_samples=size(listing)
print num_samples

for file in listing:
    im = Image.open(path1 + '/' + file) 
    img = im.crop((x1,y1,x2,y2))  #crop the images to x1-x2 x y1-y2 centered on tumor
    #img = im.resize((img_rows,img_cols))
    gray = img.convert('RGB')  #gray is a misnormer since we are converting to RGB        
    gray.save(path2 +'/' +  file, "PNG")

imlist = os.listdir(path2)
imlist.sort() #make sure to SORT the images to match 

im1 = array(Image.open( path2+ '/'+ imlist[0])) # open one image to get size
m,n = im1.shape[0:2] # get the size of the images
imnbr = len(imlist) # get the number of images

#create matrix to store all flattened images
immatrix = array([array(Image.open(path2+ '/' + im2)).flatten()
              for im2 in imlist],'f')             
              
#define labels as the outcomes#

Outcomes = pd.read_csv(Outcomes_file) #outcomes file is sorted so as to match the image index
Outcome_of_interest = pd.Series.as_matrix(Outcomes.loc[:,'FAKEHIST']) #FAKEHIST labels of interest
PID = pd.Series.as_matrix(Outcomes.loc[:,'PID'])

label = Outcome_of_interest

data,Label = shuffle(immatrix,label, random_state = 2)
#data,Label = immatrix,label

train_data = [data,Label]
(X, y) = (train_data[0],train_data[1])
X = X.reshape(X.shape[0],  img_rows, img_cols,3)
X = X.astype('float32')

## This pre-processing step changes performamce
#X /= 255 
X = preprocess_input(X)

## Extract Deep features ##

features = deep_model.predict(X)  #consider changing this from just train
Features = np.squeeze(features)

## Select the best features ##
#F = SelectKBest(f_classif, k= num_best_features).fit_transform(Features, y) # Using f_classif
#F = SelectKBest(chi2, k= num_best_features).fit_transform(Features, y) # using chi2

# Use lasso for feature selection
#llas = linear_model.Lasso(alpha= 0.01).fit(Features, y)
#feature_model = SelectFromModel(llas, prefit=True)
#F = feature_model.transform(Features)

# Use principal component analysis for best feature selection

random.seed(10) # set random starting point 
pca = PCA(n_components = num_best_features)
pcam = pca.fit(Features,y)
F = pcam.transform(Features)

plt.figure(1)
plt.plot(np.cumsum(pcam.explained_variance_ratio_))
plt.xlabel('Principal Component')
plt.ylabel('Cumulative Explained Variance');


llas = linear_model.Lasso(alpha= 0.1).fit(F, y)
feature_model = SelectFromModel(llas, prefit=True)
F = feature_model.transform(F)

# split F and y into training and testing sets

F_train, F_test, y_train, y_test = train_test_split(F, y, test_size=test_size) #use non random data splitting 
## Run best features on the Machine learning classifier model ##
x_train = F_train#X_new_train
y_train = y_train

x_test = F_test #X_new_test
y_test = y_test

#Define model: uncomment the model of interest below

#ols = linear_model.Lasso(alpha =0.1) #LASSO
#ols = RandomForestClassifier() # Random Forrest
#ols = svm.LinearSVC()  #Support vector machine, also try SVC! 
#ols = svm.SVC(kernel="linear", C=0.1)
ols = KNeighborsClassifier(n_neighbors=5) # paper had k = 4. 

ml_model = ols.fit(x_train, y_train) #define machine learning model

y_pred = ml_model.predict(x_test)
y_pred[y_pred<0] = 0 #make sure predictions are never negative (i.e. assign these to 0)
y_pred[y_pred>1] =1 #make sure predictions are never greater than 1. I.e assign these to 1. 
print(y_pred[0:8])
## Evaluate Model Performance ##
def AUC(test_labels,test_prediction,nb):

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(nb):
        # ( actual labels, predicted probabilities )
        fpr[i], tpr[i], _ = roc_curve(test_labels[:, i], test_prediction[:, i] ) # flip here
        roc_auc[i] = auc(fpr[i], tpr[i])
    return [ round(roc_auc[x],3) for x in range(nb) ] 

nb_classes = 2
y_test2 = np_utils.to_categorical(y_test, nb_classes)
y_pred2 = np.row_stack([1-abs(y_pred), y_pred]).T
y_pred2 = abs(y_pred2)


acc = accuracy_score(np.asarray(y_test), np.round(y_pred))
#acc = sum(y_pred == y_test)/len(y_test)
print('Accuracy:', acc)

ROC2 = AUC(y_test2, y_pred2, nb_classes)
print('AUC:', ROC2[1]) #AUC 


def AUCalt( test_labels , test_prediction):
    # convert to non-categorial
    test_prediction = np.array( [ x[1] for x in test_prediction   ])
    test_labels = np.array( [ 0 if x[0]==1 else 1 for x in test_labels   ])
    # get rates
    fpr, tpr, thresholds = roc_curve(test_labels, test_prediction, pos_label=1)
    # get auc
    myAuc = plt.plot(fpr, tpr)
    plt.xlabel('1-Sensitivity')
    plt.ylabel('Specificity')
    plt.title('ML Classifier ROC Curve')
    plt.grid(True)
    plt.legend(['Classifier: '],loc=4)
    plt.show()
    return myAuc
   
plt.figure(2)    
ROC_LASSO = AUCalt(y_test2, y_pred2)
			
    	
