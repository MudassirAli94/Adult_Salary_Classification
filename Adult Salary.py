
# coding: utf-8
# Dataset: https://www.kaggle.com/uciml/adult-census-income#adult.csv

# Objective: 

# 1) Perform data visualization on the variables most affected by the salary and interpert the graphs.

# 2) Perform classification on the salary column to try to get an adequate accuracy.
# In[1]:


# General Packages
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import time

# General Mathematics package
import math as math

# Graphing Packages
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use("ggplot")

# Statistics Packages
from scipy.stats import randint
from scipy.stats import skew

# Machine Learning Packages
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.preprocessing import scale
from sklearn import preprocessing
from skimage.transform import resize
import xgboost as xgb

# Neural Network Packages
from keras.utils import np_utils
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
import tensorflow as tf

# H2o packages
import h2o
from h2o.automl import H2OAutoML


# In[4]:


adult = pd.read_csv("adult.data.csv")
adult.head()


# In[119]:


adult.info()


# In[118]:


adult.iloc[:,1] = pd.Categorical(adult.iloc[:,1])
adult.iloc[:,3] = pd.Categorical(adult.iloc[:,3])
adult.iloc[:,4] = pd.Categorical(adult.iloc[:,4])
for n in range(5,9):
    adult.iloc[:,n] = pd.Categorical(adult.iloc[:,n])

adult.iloc[:,-1] = pd.Categorical(adult.iloc[:,-1])
adult.iloc[:,-2] = pd.Categorical(adult.iloc[:,-2])


# In[7]:


adult.replace("?" , "Unknown" , inplace = True)


# In[59]:


pd.crosstab(adult.workclass , adult.salary).plot.bar(figsize = (20,7) , color = "cm")
plt.xticks(rotation = 360, size = 15)
plt.yticks(size = 15)
plt.xlabel("Work Class" , size = 15)
plt.ylabel("Count" , size = 15)
plt.legend(loc=2, prop={'size': 15})
plt.title("Salary of Working Class" , size = 20)
plt.show()

# As we can see from the graph those who were privately employed accounted for majority of the salary.
# In[98]:


pd.crosstab(adult.sex , adult.salary).plot.bar(figsize = (20,7) , color = "cm")
plt.xticks(rotation = 360, size = 15)
plt.yticks(size = 15)
plt.xlabel("Sex" , size = 20)
plt.ylabel("Count" , size = 15)
plt.legend(loc=2, prop={'size': 15})
plt.title("Salary based on Sex" , size = 20)
plt.show()

# Based on the graph we can see the men who make more than 50k is almost the same amount as the female that make less or equal to 50k
# In[94]:


pd.crosstab(adult.race , adult.salary).plot.bar(figsize = (20,7) , color = "cm")
plt.xticks(rotation = 360, size = 15)
plt.yticks(size = 15)
plt.xlabel("Race" , size = 15)
plt.ylabel("Count" , size = 15)
plt.legend(loc=2, prop={'size': 15})
plt.title("Salary based on Race" , size = 20)
plt.show()

# The caucasian race has more people making money than all the races combined.
# In[69]:


adult.education.value_counts()


# In[68]:


adult.education.replace(["Preschool" , "1st-4th" , "5th-6th" , "7th-8th" , "9th" , "10th" , "11th" , "12th"] ,
                        ["Elementary School" , "Elementary School" , 'Elementary School' , "Junior High School" , 
                         "High School" , "High School" , "High School" , "High School"] , inplace = True)


# In[78]:


pd.crosstab(adult.education , adult.salary).plot.bar(figsize = (20,7) , color = "cm")
plt.xticks(rotation = 60, size = 15)
plt.yticks(size = 15)
plt.xlabel("Education" , size = 15)
plt.ylabel("Count" , size = 15)
plt.legend(loc=2, prop={'size': 15})
plt.title("Salary based on Education" , size = 20)
plt.show()

# Those who completed high school had the most people making less than or equal to 50k salary.
# Those who completed their bachelors had the most people making more than a 50k salary.
# In[87]:


pd.crosstab(adult["marital-status"] , adult.salary).plot.bar(figsize = (20,7) , color = "cm")
plt.xticks(rotation = 30, size = 15)
plt.yticks(size = 15)
plt.xlabel("Marital-Status" , size = 15)
plt.ylabel("Count" , size = 15)
plt.legend(loc=2, prop={'size': 15})
plt.title("Salary based on Marital-Status" , size = 20)
plt.show()

# Those who were never married accounted for the most in the salary for making 50k or less.
# Those who were married accounted for the most in the salary for making more than 50k.
# In[102]:


pd.crosstab(adult.relationship , adult.salary).plot.bar(figsize = (20,7) , color = "cm")
plt.xticks(rotation = 360, size = 15)
plt.yticks(size = 15)
plt.xlabel("Relationship" , size = 15)
plt.ylabel("Count" , size = 15)
plt.legend(loc=1, prop={'size': 15})
plt.title("Salary based on Relationship" , size = 20)
plt.show()

# Those who were not in a family accounted for the salary of 50k or less.
# The ones who are/were a husband accounted for the most in the salary of more than 50k.
# In[108]:


plt.figure(figsize = (10,10))
sns.heatmap(adult.corr() , square = True , annot = True , 
            linewidths=.8 , cmap="YlGnBu")
plt.title("Correlation Map")
plt.show()

# Majority of the variables have a very weak correlation with each other.Now that I am done with the data visualization
# I can now perform the machine leanring to do classification prediction on the salary column.
# In[124]:


adult.head()

# Education number and education represent the same thing so we can drop the education column
# In[114]:


adult.drop("education" , axis = 1 , inplace = True)


# In[126]:


le = preprocessing.LabelEncoder()
for n in range(4,9):
    adult.iloc[:,n] = le.fit_transform(adult.iloc[:,n])
adult.iloc[:,1] = le.fit_transform(adult.iloc[:,1])
adult.iloc[:,-1] = le.fit_transform(adult.iloc[:,-1])
adult.iloc[:,-2] = le.fit_transform(adult.iloc[:,-2])

adult.iloc[:,1] = pd.Categorical(adult.iloc[:,1])
adult.iloc[:,3] = pd.Categorical(adult.iloc[:,3])
adult.iloc[:,4] = pd.Categorical(adult.iloc[:,4])
for n in range(5,9):
    adult.iloc[:,n] = pd.Categorical(adult.iloc[:,n])

adult.iloc[:,-1] = pd.Categorical(adult.iloc[:,-1])
adult.iloc[:,-2] = pd.Categorical(adult.iloc[:,-2])

# Now that we have properly cleaned the data, we can begin the machine learning
# In[131]:


X_salary = adult.drop(["salary"] , axis = 1)
y_salary = adult["salary"]

X_train, X_test, y_train, y_test = train_test_split(X_salary, y_salary, test_size = 0.3, random_state=42)

# Logistic Regression: Predicting probability of the salary
# In[133]:


logreg = LogisticRegression()
logreg.fit(X_train , y_train)
y_pred = logreg.predict(X_test)

#print(confusion_matrix(y_val,y_pred))
print(classification_report(y_test , y_pred))
print("accuracy",round(accuracy_score(y_test , y_pred)*100,2),"%")

# Decision Tree: Predicting probability of the salary
# In[134]:


tree = DecisionTreeClassifier()

tree.fit(X_train , y_train)
y_pred_DT = tree.predict(X_test)
#print(confusion_matrix(y_val,y_pred))
print(classification_report(y_test , y_pred_DT))
print("accuracy:",round(accuracy_score(y_test , y_pred_DT)*100,2) , "%")

# Decision Tree + GridSearchCV: : Predicting probability of the salary
# In[135]:


param_dist = {'max_depth' : [3,None],
             'max_features' : np.arange(1,9),
             'min_samples_leaf':np.arange(1,9),
             "criterion" :["gini" , "entropy"]}
tree = DecisionTreeClassifier()
tree_cv = GridSearchCV(tree , param_dist , cv = 10 , verbose = 1)
tree_cv.fit(X_train , y_train)
y_pred_RF = tree_cv.predict(X_test)
#print(confusion_matrix(y_val,y_pred))
print(classification_report(y_test , y_pred_RF))
print("accuracy:",round(accuracy_score(y_test , y_pred_RF)*100,2),"%")

# Decision Tree + RandomizedSearchCV: Predicting probability of the salary
# In[136]:


param_dist = {'max_depth' : [3,None],
             'max_features' : np.arange(1,9),
             'min_samples_leaf':np.arange(1,9),
             "criterion" :["gini" , "entropy"]}
tree = DecisionTreeClassifier()
tree_cv = RandomizedSearchCV(tree , param_dist , cv = 10 , verbose = 1)
tree_cv.fit(X_train , y_train)
y_pred_RF = tree_cv.predict(X_test)
#print(confusion_matrix(y_val,y_pred))
print(classification_report(y_test , y_pred_RF))
print("accuracy:",round(accuracy_score(y_test , y_pred_RF)*100,2),"%")

# SVC: Predicting probability of the salary
# In[137]:


warnings.filterwarnings("ignore")

clf = SVC()

clf.fit(X_train , y_train)

y_pred_SVC = clf.predict(X_test)

print(classification_report(y_test , y_pred_SVC))
print('accuracy:' , round(accuracy_score(y_test , y_pred_SVC)*100,2),"%")

# XGBoost: Predicting probability of the salary
# In[142]:


adult_int = adult[0:-1].astype("int64")

X_boost = adult_int.drop(["salary"] , axis = 1)
y_boost = adult_int["salary"]

X_train_int, X_test_int, y_train_int, y_test_int = train_test_split(X_boost, y_boost, test_size = 0.3, random_state=42)


# In[143]:


warnings.filterwarnings("ignore")

xg = xgb.XGBClassifier(objective='reg:logistic', n_estimators = 10, seed=1234)
xg.fit(X_train_int, y_train_int)

y_pred_XGB = xg.predict(X_test_int)

print("accuracy:",round(accuracy_score(y_test_int, y_pred_XGB)*100,2) , "%")

The best accuracy obtained was by decision tree + randomized search with 84.87%