### This script allows for deep feature extraction using high performing pretrained
###Survival or Histology models. Cluster and statistical analysis is then performed using k-means and chi-square
### Author: Tafadzwa Chaunzwa 3/12/18 (happy birthday Watson!)

#KERAS
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam
from keras.utils import np_utils
from keras.models import load_model
from keras import backend as K
from keras import backend as K #this just makes stuff work
K.set_image_dim_ordering('th')
from keras.applications.vgg16 import preprocess_input

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
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix

#from other libraries
from numpy import *
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import chisquare
import scipy.stats as stats

### DATA ###
#input image dimensions
OG_size = 150
img_rows, img_cols = 50, 50 # 50, 50 #normally. make sure this is an even number
Center = OG_size/2
x1, y1, x2, y2 = Center-img_rows/2,Center-img_rows/2,Center+img_cols/2,Center+img_cols/2 # 50, 50, 100, 100

# number of channels
img_channels = 1
# data
#Outcomes_file = '/home/chintan/Desktop/MLPlayGround/Augmented_Set_Exp/TestOutcomes.csv' #define the outcomes file, sorted according to PID
#path1 = '/home/chintan/Desktop/MLPlayGround/Augmented_Set_Exp/TestImages' #change this to only test images    
#path2 = '/home/chintan/Desktop/MLPlayGround/Augmented_Set_Exp/TestImageCrops'  #DELETE  

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
Outcome_of_interest = pd.Series.as_matrix(Outcomes.loc[:,'SURV2']) #AGE70/HIST/SURV2/GENDER/STAGEITYPE/SMKSTS/PROGRESS
PID = pd.Series.as_matrix(Outcomes.loc[:,'PID'])

label = Outcome_of_interest

data,Label = immatrix,label
train_data = [data,Label]

print (train_data[0].shape)
print (train_data[1].shape)

(X, y) = (train_data[0],train_data[1])
X = X.reshape(X.shape[0], img_rows, img_cols,3)
X= X.astype('float32')

X /= 255
#X = preprocess_input(X)


print('X shape:', X.shape)
print(X.shape[0], 'test samples')


### MODEL###

## load pretrained model ##
predDir = '/home/chintan/Desktop/FinalModels'
modelFile = (os.path.join(predDir,'VGG_SurvivalModel_RightCensored_run2_BESTMODEL_2_6_18.h5')) #survival based extractor
#modelFile = (os.path.join(predDir,'VGG_Histopath_run6.h5')) #Histology based extractor 
model = load_model(modelFile)

### Extract features from layer M ###

layer_index = 19 # Set to 19 for 512-D vector, 20 for 4096-D
func1 = K.function([ model.layers[0].input , K.learning_phase()  ], [ model.layers[layer_index].output ] )

Feat = np.empty([1,1,512]) #when layer_index =19
#Feat = np.empty([1,1,4096]) # when layer_index =20
for i in xrange(X.shape[0]):
	input_image=X[i,:,:,:].reshape(1,50,50,3)
	input_image_aslist= [input_image]
	func1out = func1(input_image_aslist)
	features = np.asarray(func1out)
	Feat = np.concatenate((Feat, features), axis = 1)

Feat = squeeze(Feat)
Features = Feat[1:Feat.shape[1],:]

### Cluster Analysis ###

#1 Dimension reduction with PCA ##
num_best_features = 3  # number of components after dim reduction

#random.seed(10) # set random starting point . 3 and 10 is best
pca = PCA(n_components = num_best_features)
pcam = pca.fit(Features,y)
F = pcam.transform(Features)

#2 K means clustering #
n_clusters = 4

# Number of clusters
kmeans = KMeans(n_clusters= n_clusters)
# Fitting the input data
kmeans = kmeans.fit(F) #or just X
# Getting the cluster labels
cluster_labels = kmeans.predict(F) # careful with these "labels" very different from "label" above
# Centroid values
centroids = kmeans.cluster_centers_   
C = centroids
# X coordinates of centroids
C_x = centroids[:,0]
# Y coordinates of centroids
C_y = centroids[:,1]

# Getting the values and plotting it
f1 = F[:,0] # these are the 2D features
f2 = F[:,1]

# Plotting along with the Centroids
plt.figure()
plt.scatter(f1, f2, c= y) #color them according to cluster (cluster_labels) or outcome e.g. histology (y)
plt.scatter(C_x, C_y, marker='*', s=200, c='g')
plt.title('Clinical parameter with Centroids (k=4)')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# Get performance metrics #
cm = confusion_matrix(y, cluster_labels)  # first variable Label is vertical, second (labels) is horizontal

cluster1 = cm[:,0].astype(float)
cluster2 = cm[:,1].astype(float)
#cluster3 = cm[:,2].astype(float) #comment out or add clusters as you change value of k
#cluster4 = cm[:,3].astype(float)

# Cluster proportions #
prop1 = cluster1[0]/sum(cluster1)
print ('cluster1 0s perc: ' '%.2f' %prop1)

prop2 = cluster2[0]/sum(cluster2)
print ('cluster2 0s perc: ' '%.2f' %prop2)

#prop3 = cluster3[0]/sum(cluster3)
#print ('cluster3 0s perc: ' '%.2f' %prop3) 

#prop4 = cluster4[0]/sum(cluster4)
#print ('cluster4 0s perc: ' '%.2f' %prop4)

# Eval statistical significance #
t = cm[0:max(y)+1,0:n_clusters]
chi2, p, dof, expected = stats.chi2_contingency(t)
print ("Chi-Statistics:", chi2)
print("pvalue:", p)

