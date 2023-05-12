import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import missingno as msno
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')


# Need to clean data: First convert .data file to .csv file by right-clicking hepatitis.csv file in google colab and changing extension from .data to .csv.
# The provided file does not have column headers with attribute names so we will add them using the following code (could also do it manually in excel):
df1 = pd.read_csv(r"hepatitis.csv", header=None)

df1.rename(columns={0: 'LIVE', 1: 'AGE', 2: 'FEMALE', 3: 'STEROID', 4: 'ANTIVIRALS', 5: 'FATIGUE', 6: 'MALAISE', 7:
    'ANOREXIA', 8: 'LIVER BIG', 9: 'LIVER FIRM', 10: 'SPLEEN PALPABLE', 11: 'SPIDERS', 12: 'ASCITES', 13: 'VARICES',
                    14: 'BILIRUBIN', 15: 'ALK PHOSPHATE', 16: 'SGOT', 17: 'ALBUMIN', 18: 'PROTIME', 19: 'HISTOLOGY'},
           inplace=True)
# save to new csv file
df1.to_csv(r"hepatitis_with_col.csv", index=False)

#Read the dataset/s and mark/identify missing values (in this dataset marked by a '?')
df2 = pd.read_csv('hepatitis_with_col.csv', na_values='?')

# The dataset used strange binary (1= False, 2= True), will change this so that 0 = False, and 1 = True (just replace all 1s with 0s, and all 2s with 1s)
cols_to_replace = ['LIVE','FEMALE', 'STEROID', 'ANTIVIRALS', 'FATIGUE', 'MALAISE', 'ANOREXIA',
                   'LIVER BIG', 'LIVER FIRM', 'SPLEEN PALPABLE', 'SPIDERS', 'ASCITES', 'VARICES', 'HISTOLOGY']
for col in cols_to_replace:
    df2[col] = df2[col].replace({1: 0, 2: 1})

# Turn cleaned dataset into a new file and assign that file as our main datafile
df2.to_csv('hepatitis_cleaned.csv', index=False)

df = pd.read_csv(r"hepatitis_cleaned.csv")

# Checking first 5 rows of dataset (can also see our columns succesfully implemented and binary is fixed)
print(df.head())

# Check final 5 rows of data
print(df.tail())

# Check shape of dataset (no. of rows, no. of columns)
print(df.shape)

# Check how many unique variable there are per variable.
print(df.nunique())

# Get generalized information about dataset (column names, non-null values, column type)
print(df.info())

# Display how many missing values there are per attribute.
missing_values = df.isnull().sum()
print(missing_values)

# Need to change the dType of our columns/variables. In initial dataset, all variables that had missing values were cateogrised as objects.
#  We will change our categorical variables to 'Int64' type. For our continous variables, we will change them to 'Float64' types.
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

# Check to see if the types succesfully updated.
print(df.info())

# Check the overall distribution of our categorical variables (gives us useful information i.e. most patients on antivirals, most patients anorexic, etc.)
categorical_variables = ['FEMALE', 'STEROID', 'ANTIVIRALS', 'FATIGUE', 'MALAISE','ANOREXIA', 'LIVER BIG', 'LIVER FIRM','SPLEEN PALPABLE', 'SPIDERS', 'ASCITES','VARICES', 'HISTOLOGY']
df[categorical_variables].apply(pd.Series.value_counts)

# Create plot that shows relationship between age and 'LIVE' allowing us to observe if there is any clear relationship between age and survival
plt.figure(figsize=(20, 6))
sns.countplot(x='AGE', hue='LIVE', data=df)

# Create plot to observe number of males in dataset vs females
plt.figure(figsize=(20, 6))
male_female = sns.countplot(x='FEMALE', data=df)
plt.gca().set_xticklabels(['Male','Female']);
plt.show()

# Create plot to observe survival between men and women
plt.figure(figsize=(20, 6))
male_female_survival = sns.countplot(x='LIVE', hue='FEMALE', data=df)
plt.gca().set_xticklabels(['Death','Alive']);
plt.legend(title='SEX', loc='upper left', labels=['Male', 'Female'])
plt.show()

# Create plots for all variables #
fig = plt.figure(figsize =(18,18))
ax=fig.gca()
df.hist(ax=ax,bins =30)
plt.show()

# Create plots for each variable showing outliers
df.plot(kind='box', subplots=True,
        layout=(5,4),sharex=False,sharey=False, figsize=(20, 10), color='deeppink');

# Identify the outliers among our continuous variables
continous_features = ['AGE','BILIRUBIN','ALK PHOSPHATE','SGOT','ALBUMIN', 'PROTIME']

def outliers(df_out, drop = False):
    for each_feature in df_out.columns:
        feature_data = df_out[each_feature]
        Q1 = feature_data.quantile(0.25) # 25th percentile of the data of the given feature
        Q3 = feature_data.quantile(0.75) # 75th percentile of the data of the given feature
        IQR = Q3-Q1 #Interquartile Range
        outlier_step = IQR * 1.5
        mask = (feature_data >= Q1 - outlier_step) & (feature_data <= Q3 + outlier_step)
        mask = mask | feature_data.isna()  # Include NaN values in the mask
        outliers = feature_data[~mask].index.tolist()
        if not drop:
            print('For the feature {}, No of Outliers is {}'.format(each_feature, len(outliers)))
        if drop:
            df_out.drop(outliers, inplace = True, errors = 'ignore')
            print('Outliers from {} feature removed'.format(each_feature))

# Display number of outliers for each variable
outliers(df[continous_features])

# We could choose to drop the outliers using the following:
# outliers(df[continous_features], drop = True)
# However, I am not going to do this because we are already dealing with miss ing values and a relatively small amount of instances (155).
#By removing outliers, we would make the already small data size smaller.


# Create plot to see how many people lived vs died (0 = died, 1 = lived) - allows us to observe broad survival likelihood
# From this can gather that an individual with hepatitis is much more likely to 'live' than 'die'
print(df.LIVE.value_counts())
fig, ax = plt.subplots(figsize=(5,4))
name = ["Live", "Die"]
ax = df.LIVE.value_counts().plot(kind='bar')
ax.set_title("Hepatitis Survival Classes", fontsize = 13, weight = 'bold')
ax.set_xticklabels (name, rotation = 0)

# To calculate and show the percentage
totals = []
for i in ax.patches:
    totals.append(i.get_height())
total = sum(totals)
for i in ax.patches:
    ax.text(i.get_x()+.09, i.get_height()-50, \
            str(round((i.get_height()/total)*100, 2))+'%', fontsize=14,
                color='white', weight = 'bold')

# Create table showing correlation between variables
sns.set(style="white")
plt.rcParams['figure.figsize'] = (30, 30)
sns.heatmap(df.corr(), annot = True, linewidths=.5, cmap="Blues")
plt.title('Corelation Between Variables', fontsize = 30)
plt.show()

