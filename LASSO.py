import pandas as pd
from sklearn import linear_model
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score

from keras.utils import np_utils
import numpy as np
import matplotlib.pyplot as plt



dataTrain = pd.read_csv("TrainO_RightCensored.csv")
dataTest = pd.read_csv("TestO_RightCensored.csv")
# print df.head()

x_train = dataTrain[['SEX', 'AGE', 'HISTOLOGY', 'STAGEITYPE', 'SMOKSORT']] 
y_train = dataTrain['SURV2']

x_test = dataTest[['SEX', 'AGE', 'HISTOLOGY', 'STAGEITYPE', 'SMOKSORT']] 
y_test = dataTest['SURV2']

ols = linear_model.Lasso(alpha =0.1)
model = ols.fit(x_train, y_train)

print model.predict(x_test)[0:10]

y_pred = model.predict(x_test)


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

y_test2 = pd.Series.as_matrix(dataTest.loc[:,'SURV2'])
y_test2 = np_utils.to_categorical(y_test2, nb_classes)

y_pred2 = np.row_stack([1-y_pred, y_pred]).T

ROC2 = AUC(y_test2, y_pred2, nb_classes)

print('AUC:', ROC2[1]) #AUC 

acc = accuracy_score(np.asarray(y_test), np.round(y_pred), normalize=False)

print('Accuracy:', acc)


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
    plt.title('Multivariate CPF regression ROC Curve')
    plt.grid(True)
    plt.legend(['Lasso'],loc=4)
    plt.show()
    return myAuc
   
plt.figure(1)    
ROC_LASSO = AUCalt(y_test2, y_pred2)
