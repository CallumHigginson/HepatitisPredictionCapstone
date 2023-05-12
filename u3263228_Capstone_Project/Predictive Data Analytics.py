# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
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







# Identify the outliers among our continuous variables
continous_features = ['AGE','BILIRUBIN','ALK PHOSPHATE','SGOT','ALBUMIN', 'PROTIME']

df = pd.read_csv('hepatitis_cleaned.csv')

# First we will create a copy of our dataset. This copy will become our dataset that has no missing values.
hepatitis_no_missing = df.copy()
# Specify the attribute to be predicted (in our case, LIVE)
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


# To continue with the data visualisation and task in general, we need to deal with these missing values. As we can see above, the columns ALK PHOSPHATE, ALBUMIN, BILIRUBIN, SGOT, and PROTIME columns have missing values.
# There are numerous ways we could approach this missing data problem. All the rows with missing data could be deleted, the columns that have lots of missing data (PROTIME) in this case could be deleted,
# or we could substitute the missing values with the mean, or median value of the column.
# Here I will substitute the missing values of ALK PHOSPHATE, ALBUMIN, and PROTIME with their median values.
# I chose this over using the mean since the mean could be skewed heavily by outliers. I didn't want to delete the rows as the number of instances in this dataset is already quite low (155).
# We should first find the median of these columns

median_ALK_PHOSPHATE = hepatitis_no_missing['ALK PHOSPHATE'].median()
print(f'Median of ALK_PHOSPHATE is: {median_ALK_PHOSPHATE}')

median_ALBUMIN = hepatitis_no_missing['ALBUMIN'].median()
print(f'Median of ALBUMIN is: {median_ALBUMIN}')

median_PROTIME = hepatitis_no_missing['PROTIME'].median()
print(f'Median of PROTIME is: {median_PROTIME}')

median_BILIRUBIN = hepatitis_no_missing['BILIRUBIN'].median()
print(f'Median of BILIRUBIN is: {median_BILIRUBIN}')

median_SGOT = hepatitis_no_missing['SGOT'].median()
print(f'Median of SGOT is: {median_SGOT}')

# I checked all these medians manually and confirmed that the correct values were returned.
# Now, we will fill in our missing values in these columns with the medians.
hepatitis_no_missing['ALK PHOSPHATE'].fillna(hepatitis_no_missing['ALK PHOSPHATE'].median(), inplace=True)

hepatitis_no_missing['ALBUMIN'].fillna(hepatitis_no_missing['ALBUMIN'].median(), inplace=True)

hepatitis_no_missing['PROTIME'].fillna(hepatitis_no_missing['PROTIME'].median(), inplace=True)

hepatitis_no_missing['BILIRUBIN'].fillna(hepatitis_no_missing['BILIRUBIN'].median(), inplace=True)

hepatitis_no_missing['SGOT'].fillna(hepatitis_no_missing['SGOT'].median(), inplace=True)
# Check the first five rows of these columns to see if the missing values were successfully imputed
hepatitis_no_missing[continous_features].head()

# Now we double-check if the missing value count is now zero for those columns
missing_values = hepatitis_no_missing.isnull().sum()
print(missing_values)

# Now that we have succesfully input the median values in place of the missing values in our columns with continuous values, we will now move to remove the rows with missing values in our columns that have discrete values.
# I chose to remove the rows in these cases because 1) there are quite a few less missing values in these columns
# (most being 11 in 'LIVER FIRM'), and 2) because using the median/mean would be more biased when it comes to discrete
# values (all will either be 1 or 2). The columns we will be targetting here are: STEROID, FATIGUE, MALAISE, ANOREXIA,
# LIVER BIG, LIVER FIRM, SPLEEN PALPABLE, SPIDERS, ASCITES, VARICES. We can drop the rows with missing values in these columns with the following code:
hepatitis_no_missing.dropna(subset=['STEROID', 'FATIGUE', 'MALAISE','ANOREXIA', 'LIVER BIG', 'LIVER FIRM',
                                    'SPLEEN PALPABLE', 'SPIDERS', 'ASCITES','VARICES'], inplace=True)


# We will now do a check for missing values again
missing_values = hepatitis_no_missing.isnull().sum()
print(missing_values)

# Since we dropped rows with missing data, we will use .info to see how many instances we lost (in this case, we lost 13)
hepatitis_no_missing.info()

# Observe how the dataset has changed broadly after imputing values and removing rows
hepatitis_no_missing.describe()

# Display correlation between variables once again but this time with our imputed dataset
sns.set(style="white")
plt.rcParams['figure.figsize'] = (30, 30)
sns.heatmap(hepatitis_no_missing.corr(), annot = True, linewidths=.5, cmap="Blues")
plt.title('Corelation Between Variables', fontsize = 30)
plt.show()

#pre-processing
le = preprocessing.LabelEncoder()
AGE = le.fit_transform(list(hepatitis_no_missing["AGE"])) # age in years
GENDER = le.fit_transform(list(hepatitis_no_missing["FEMALE"])) # gender (0 = male; 1 = female)
STEROID = le.fit_transform(list(hepatitis_no_missing["STEROID"])) # taking steroids
ANTIVIRALS = le.fit_transform(list(hepatitis_no_missing["ANTIVIRALS"])) # taking antivirals
FATIGUE = le.fit_transform(list(hepatitis_no_missing["FATIGUE"])) # patient reported being fatigued
MALAISE = le.fit_transform(list(hepatitis_no_missing["MALAISE"])) # patient reported being malaise
ANOREXIA = le.fit_transform(list(hepatitis_no_missing["ANOREXIA"])) # patient anorexic
LIVER_BIG = le.fit_transform(list(hepatitis_no_missing["LIVER BIG"])) # patient's liver big
LIVER_FIRM = le.fit_transform(list(hepatitis_no_missing["LIVER FIRM"])) # patient's liver firm
SPLEEN_PALPABLE = le.fit_transform(list(hepatitis_no_missing["SPLEEN PALPABLE"]))# patient's spleen palpable
SPIDERS = le.fit_transform(list(hepatitis_no_missing["SPIDERS"])) # patient has spider angiomas
ASCITES = le.fit_transform(list(hepatitis_no_missing["ASCITES"])) # patient has ascites
VARICES = le.fit_transform(list(hepatitis_no_missing["VARICES"])) # patient has esophageal varices
BILIRUBIN = le.fit_transform(list(hepatitis_no_missing["BILIRUBIN"])) # amount of bilirubin in patient's blood
ALK_PHOSPHATE = le.fit_transform(list(hepatitis_no_missing["ALK PHOSPHATE"])) # amount of ALK phosphatase in patient's blood
SGOT = le.fit_transform(list(hepatitis_no_missing["SGOT"])) # how much AST in patient's blood
ALBUMIN = le.fit_transform(list(hepatitis_no_missing["ALBUMIN"])) # how much albumin in patient's blood
PROTIME = le.fit_transform(list(hepatitis_no_missing["PROTIME"])) # patient's prothrombim time test score (seconds)
HISTOLOGY = le.fit_transform(list(hepatitis_no_missing["HISTOLOGY"])) # patient had histology done
LIVE = le.fit_transform(list(hepatitis_no_missing["LIVE"])) # Survival 2 - live; 1 – die


# Store variable values as x (arranged in proper rows using zip)
x = list(zip(AGE, GENDER, STEROID, ANTIVIRALS, FATIGUE, MALAISE, ANOREXIA, LIVER_BIG, LIVER_FIRM, SPLEEN_PALPABLE,
             SPIDERS, ASCITES, VARICES, BILIRUBIN, ALK_PHOSPHATE, SGOT, ALBUMIN, PROTIME, HISTOLOGY))
# Store LIVE values as y
y = list(LIVE)

# Set parameters for cross-validation process
import sklearn.model_selection
num_folds = 5
seed = 7
scoring = 'accuracy'


# Splitting dataset into training and testing data using train_test_split function
import sklearn.model_selection
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.20, random_state=seed)
# I opted for an 80 20 split here. I tested a 70 30 split and a 60 40 split but these splits resulted in the model’s accuracy being significantly lower.

# Check the size of the training and testing subsets
np.shape(x_train), np.shape(x_test)

# Create a list which will be used to store the models we want to use and then append the models to the list
models = []
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
models.append(('GBM', GradientBoostingClassifier()))
models.append(('RF', RandomForestClassifier()))
dt = DecisionTreeClassifier()
nb = GaussianNB()
gb = GradientBoostingClassifier()
rf = RandomForestClassifier()
results = []
names = []

# Assess performance of each model using cross validation and print results
print("Performance on Training set")
for name, model in models:
  kfold = KFold(n_splits=num_folds,shuffle=True,random_state=seed)
  cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')
  results.append(cv_results)
  names.append(name)
  msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
  msg += '\n'
  print(msg)

# From numbers returned above, store the best performing model in a variable (in our case, best performing model was RF)
best_model = rf
# Train the rf model on the dataset
best_model.fit(x_train, y_train)
# Get the model to make predictions on the test data
y_pred = best_model.predict(x_test)
# Use sklearn function to determine the accuracy_score of the model
model_accuracy = accuracy_score(y_test, y_pred)
# Create plot providing a comparison of the Algorithms' Performance
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

#Model Performance Evaluation Metric 1 - Classification Report
print(classification_report(y_test, y_pred))

#Model Performance Evaluation Metric 2 - Confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

#Model Evaluation Metric 3- ROC-AUC curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

best_model = gb
best_model.fit(x_train, y_train)
rf_roc_auc = roc_auc_score(y_test,best_model.predict(x_test))
fpr,tpr,thresholds = roc_curve(y_test, best_model.predict_proba(x_test)[:,1])

plt.figure()
plt.plot(fpr,tpr,label = 'Random Forest(area = %0.2f)'% rf_roc_auc)
plt.plot([0,1],[0,1],'r--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.savefig('LOC_ROC')
plt.show()

#Model Evaluation Metric 4- prediction report
for x in range(len(y_pred)):
  print("Predicted: ", y_pred[x], "Actual: ", y_test[x], "Data: ", x_test[x],)
