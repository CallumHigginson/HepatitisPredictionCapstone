# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.utils import shuffle
from sklearn import datasets, svm, linear_model, preprocessing
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
import pickle
from matplotlib import style





# Data file import
df = pd.read_csv('hepatitis_cleaned.csv')

hepatitis_no_missing = df.copy()
# Attribute to be predicted
predict = "LIVE"



# We can see from df.info() that the Dtype of our columns with missing values are 'object'. In this form,
# we cannot proceed with our data visualsiation. To address this, we need to change the types of our columns.
# For columns with discrete values, we will change them to 'Int64' type.
# For our columns with continuous values, we will change them to 'Float64' types.
df['STEROID'] = df['STEROID'].astype('Int64')
df['LIVER BIG'] = df['LIVER BIG'].astype('Int64')
df['FATIGUE'] = df['FATIGUE'].astype('Int64')
df['MALAISE'] = df['MALAISE'].astype('Int64')
df['ANOREXIA'] = df['ANOREXIA'].astype('Int64')
df['LIVER BIG'] = df['LIVER BIG'].astype('Int64')
df['LIVER FIRM'] = df['LIVER FIRM'].astype('Int64')
df['SPLEEN PALPABLE'] = df['SPLEEN PALPABLE'].astype('Int64')
df['SPIDERS'] = df['SPIDERS'].astype('Int64')
df['ASCITES'] = df['ASCITES'].astype('Int64')
df['VARICES'] = df['VARICES'].astype('Int64')
df['BILIRUBIN'] = df['BILIRUBIN'].astype('Float64')
df['ALK PHOSPHATE'] = df['ALK PHOSPHATE'].astype('Float64')
df['SGOT'] = df['SGOT'].astype('Float64')
df['ALBUMIN'] = df['ALBUMIN'].astype('Float64')
df['PROTIME'] = df['PROTIME'].astype('Float64')




# I checked all these medians manually and confirmed that the correct values were returned.
# Now, we will fill in our missing values with these medians.
hepatitis_no_missing['ALK PHOSPHATE'].fillna(hepatitis_no_missing['ALK PHOSPHATE'].median(), inplace=True)
hepatitis_no_missing['ALBUMIN'].fillna(hepatitis_no_missing['ALBUMIN'].median(), inplace=True)
hepatitis_no_missing['PROTIME'].fillna(hepatitis_no_missing['PROTIME'].median(), inplace=True)
hepatitis_no_missing['BILIRUBIN'].fillna(hepatitis_no_missing['BILIRUBIN'].median(), inplace=True)
hepatitis_no_missing['SGOT'].fillna(hepatitis_no_missing['SGOT'].median(), inplace=True)



# Now that we have succesfully input the median values in place of the missing values in our columns with continuous
# values, we will now move to remove the rows with missing values in our columns that have discrete values.
# I chose to remove the rows in these cases because 1) there are quite a few less missing values in these columns
# (most being 11 in 'LIVER FIRM'), and 2) because using the median/mean would be more biased when it comes to discrete
# values (all will either be 1 or 2). The columns we will be targetting here are: STEROID, FATIGUE, MALAISE, ANOREXIA,
# LIVER BIG, LIVER FIRM, SPLEEN PALPABLE, SPIDERS, ASCITES, VARICES. We can drop the rows with missing values in
# these columns with the following code:
hepatitis_no_missing.dropna(subset=['STEROID', 'FATIGUE', 'MALAISE','ANOREXIA', 'LIVER BIG', 'LIVER FIRM',
                                    'SPLEEN PALPABLE', 'SPIDERS', 'ASCITES','VARICES'], inplace=True)




le = preprocessing.LabelEncoder()
AGE = le.fit_transform(list(hepatitis_no_missing["AGE"])) # age in years
GENDER = le.fit_transform(list(hepatitis_no_missing["FEMALE"])) # gender (0 = male; 1 = female)
STEROID = le.fit_transform(list(hepatitis_no_missing["STEROID"])) # chest-pain and chest-pain type
ANTIVIRALS = le.fit_transform(list(hepatitis_no_missing["ANTIVIRALS"])) # resting blood pressure (mm/Hg)
FATIGUE = le.fit_transform(list(hepatitis_no_missing["FATIGUE"])) # serum cholestrol (mg/dl)
MALAISE = le.fit_transform(list(hepatitis_no_missing["MALAISE"])) # fasting blood sugar
ANOREXIA = le.fit_transform(list(hepatitis_no_missing["ANOREXIA"])) # resting elctrocardiographic results
LIVER_BIG = le.fit_transform(list(hepatitis_no_missing["LIVER BIG"]))
LIVER_FIRM = le.fit_transform(list(hepatitis_no_missing["LIVER FIRM"]))
SPLEEN_PALPABLE = le.fit_transform(list(hepatitis_no_missing["SPLEEN PALPABLE"]))
SPIDERS = le.fit_transform(list(hepatitis_no_missing["SPIDERS"]))
ASCITES = le.fit_transform(list(hepatitis_no_missing["ASCITES"]))
VARICES = le.fit_transform(list(hepatitis_no_missing["VARICES"]))
ALK_PHOSPHATE = le.fit_transform(list(hepatitis_no_missing["ALK PHOSPHATE"]))
BILIRUBIN = le.fit_transform(list(hepatitis_no_missing["BILIRUBIN"]))
SGOT = le.fit_transform(list(hepatitis_no_missing["SGOT"]))
ALBUMIN = le.fit_transform(list(hepatitis_no_missing["ALBUMIN"]))
PROTIME = le.fit_transform(list(hepatitis_no_missing["PROTIME"]))
HISTOLOGY = le.fit_transform(list(hepatitis_no_missing["HISTOLOGY"]))
LIVE = le.fit_transform(list(hepatitis_no_missing["LIVE"])) # Survival 2 - live; 1 - die



x = list(zip(AGE, GENDER, STEROID, ANTIVIRALS, FATIGUE, MALAISE, ANOREXIA, LIVER_BIG, LIVER_FIRM, SPLEEN_PALPABLE,
             SPIDERS, ASCITES, VARICES, BILIRUBIN, ALK_PHOSPHATE, SGOT, ALBUMIN, PROTIME, HISTOLOGY))
y = list(LIVE)
# Test options and evaluation metric
num_folds = 5
seed = 7
scoring = 'accuracy'

# Model Test/Train
# Splitting what we are trying to predict into 4 different arrays -
# X train is a section of the x array(attributes) and vise versa for Y(features)
# The test data will test the accuracy of the model created
import sklearn.model_selection
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.20, random_state=seed)
#splitting 20% of our data into test samples. If we train the model with higher data it already has seen that
# information and knows


models = []
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
models.append(('GBM', GradientBoostingClassifier()))
models.append(('RF', RandomForestClassifier()))
results = []
names = []

for name, model in models:
  kfold = KFold(n_splits=num_folds,shuffle=True,random_state=seed)
  cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')
  results.append(cv_results)
  names.append(name)
  msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
  msg += '\n'
  print(msg)




#Model Evaluation by testing with independent/external test data set.
# Make predictions on validation/test dataset

models.append(('DT', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
models.append(('GBM', GradientBoostingClassifier()))
models.append(('RF', RandomForestClassifier()))
dt = DecisionTreeClassifier()
nb = GaussianNB()
gb = GradientBoostingClassifier()
rf = RandomForestClassifier()

best_model = rf
best_model.fit(x_train, y_train)
y_pred = best_model.predict(x_test)
model_accuracy = accuracy_score(y_test, y_pred)


#Model Evaluation Metric 4-prediction report
for x in range(len(y_pred)):
  print("Predicted: ", y_pred[x], "Actual: ", y_test[x], "Data: ", x_test[x],)
