
import pandas as pd
df = pd.read_csv("breast-cancer-wisconsin-data.csv") 
df

df.shape
list(df)
###########################################################################
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Class'] = le.fit_transform(df[ 'diagnosis'])
df['Class']
list(df)
###########################################################################

X = df.iloc[:,2:32]
X
list(X)

Y = df['Class']
Y
###########################################################################
from sklearn.model_selection import train_test_split
pd.options.display.float_format = '{:.2f}%'.format
X_train,X_test,Y_train,Y_test  = train_test_split(X,Y, test_size=0.30, random_state = 150, stratify = Y)
print(X_train.shape,X_test.shape,Y_train.shape,Y_test.shape)


pd.crosstab(Y,Y,normalize='all')
pd.crosstab(Y_train,Y_train,normalize='all')
pd.crosstab(Y_test,Y_test,normalize='all')
###########################################################################

from sklearn.linear_model import LogisticRegression
log_re = LogisticRegression()
log_re.fit(X_train,Y_train)

y_pred = log_re.predict(X_test)
y_pred

from sklearn import metrics
cm = metrics.confusion_matrix(y_pred,Y_test)
cm
metrics.accuracy_score(y_pred,Y_test)
###########################################################################

from sklearn.metrics import log_loss
loss = log_loss(y_pred,Y_test)
print("Log loss / cross entropy = ", loss.round(2))
###########################################################################

from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import GridSearchCV


alpha_val={'alpha': [0.001,0.002,0.003,0.004,0.005,0.009,0.01,0.02,0.03,0.04,0.05,0.06,0.1,0.2,0.3,0.4,0.5,1,2,3,5,10,20,50,100]}

ridge_cv=GridSearchCV(RidgeClassifier(),alpha_val,scoring='accuracy',cv=10)
ridge_cv

#r_m1=ridge_cv.fit(X_scale,Y)
#print(r_m1.best_score_)#--->0.961
#print(r_m1.best_params_)#--->10

r_m1=ridge_cv.fit(X_train,Y_train)
print(r_m1.best_score_)#--->0.959
print(r_m1.best_params_)#--->1

r_m2=RidgeClassifier(alpha=1)
r_m2.fit(X_train,Y_train)
r_m2.intercept_
r_m2.coef_

pred_train=r_m2.predict(X_train)
print ("Training log loss", log_loss(Y_train,pred_train).round(3))#-->1.041
print ("Training Accuracy", metrics.accuracy_score(Y_train,pred_train).round(3))#-->0.97

pred_test = r_m2.predict(X_test)
print ("Test log loss", log_loss(Y_test,pred_test).round(3))#-->1.414
print ("Test Accuracy", metrics.accuracy_score(Y_test,pred_test).round(3))#-->0.959

plt.figure(figsize = (12,6))
plt.axhline(0,color='red',linestyle='solid')
plt.plot(range(len(X.columns)), r_m2.coef_[0])

plt.show()

import numpy as np
coeff_estimator = pd.concat([pd.DataFrame(X.columns), pd.DataFrame(np.transpose(r_m2.coef_))] ,axis =1, ignore_index=True)
coeff_estimator

coeff_estimator.sort_values(by=1)

##########################################################################
##Logistic Regression:

X1=df[df.columns[[28,21,7,29,22,6,10,20]]]
list(X1)

X_train,X_test,Y_train,Y_test=train_test_split(X1,Y, test_size=0.30, random_state=70, stratify=Y)

lr2=LogisticRegression()
lr2.fit(X_train,Y_train)

Y_pred2=lr2.predict(X_test)
Y_pred2

cm=metrics.confusion_matrix(Y_test,Y_pred2)
cm

metrics.accuracy_score(Y_pred2,Y_test).round(3)#-->0.953

l_loss2=log_loss(Y_pred2,Y_test)
print("Log loss -> ", l_loss2.round(2))#-->1.62
########################################################
##KNN:

X1=df[df.columns[[28,21,7,29,22,6,10,20]]]
list(X1)

X_train,X_test,Y_train,Y_test=train_test_split(X1,Y,test_size=0.30,random_state=70)

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train,Y_train)

Y_pred_knn=knn.predict(X_test)

cm1=metrics.confusion_matrix(Y_test,Y_pred_knn)
cm1

knn_acc = metrics.accuracy_score(Y_test,Y_pred_knn)
print("Accuracy for k=7",knn_acc)#-->0.935
##########################################################################

from sklearn.model_selection import GridSearchCV

k=range(1,51)
P_val=[1,2]

hyperparameters=dict(n_neighbors =k, p=P_val)
clf=GridSearchCV(KNeighborsClassifier(),hyperparameters,cv=10)

knn1=clf.fit(X_train,Y_train)

knn1.cv_results_

print(knn1.best_score_) #--------------->93.4

print(knn1.best_params_)#-------------------->k=15 p=1
##############################
'''based on the results from logistic and KNN
we can conclude that logistic regression gets best accuracy score than KNN
logistic reg-accuracy score=95.3
KNN classifier -best score =93.4'''

