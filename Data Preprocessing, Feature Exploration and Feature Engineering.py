#!/usr/bin/env python
# coding: utf-8

# # Pre-Modeling: Data Preprocessing and Feature Exploration in Python

# ## Goal

# - Goal:
#     - Pre-modeling/modeling 80%/20% of work
#     - Show the importance of data preprocessing, feature exploration, and feature engineering on model performace
#     - Go over a few effective pre-modeling steps
#     - This is only a small subset of pre-modeling 
# 
# - Format:
#     - Tutorial style 
#     - Walk through concepts and code (and point out libraries)
#     - Use an edited version of the 'adult' dataset (to predict income) with the objective of building a binary classification model
#   
#   
# - Python libraries:
#     - Numpy
#     - Pandas
#     - Sci-kit learn
#     - Matplotlib
#     - Almost entire workflow is covered by these four libraries
# 
# Source of 'adult' dataset: https://archive.ics.uci.edu/ml/datasets/Adult

# ## Agenda

# 
# 1. Modeling Overview 
# 2. Introduce the Data
# 3. Basic Data Cleaning
#     1. Dealing with data types
#     2. Handling missing data
# 4. More Data Exploration
#     1. Outlier detection
#     2. Plotting distributions
# 5. Feature Engineering
#     1. Interactions between features
#     2. Dimensionality reduction using PCA
# 6. Feature Selection and Model Building

# ## Part 1: Modeling Overview

# ### Review of predictive modeling

# - **Definition**
#     - Statistical technique to predict unknown outcomes
#     - Today's example
#         - Binary classification model - determine the probability that an observation belongs to one of two groups
#         - Examples:
#             - Whether a person votes for one of two political candidates
#             - Whether a credit card transaction is fraud
#             - Whether or not a person will be diagnosed with a given disease in the next year
# 
# 
# - **Data terminology**
#     - Inputs - Independent variables (also called features)
#         - Predictors
#     - Outputs - Dependent variable (also called the outcome)
#         - The target variable for prediction
#     - Models explain the effect that features have on the outcome
#         
#         
# - **Assessing model performance**
#     - Randomly split observations into train/test sets
#     - Build model on train set and assess performance on test set
#     - AUC of ROC is common performance metric
#         - True positive vs. false positive rates
# 
# 
# - **Types of models for binary classification**
#     - Logistic regression
#     - Random Forest
#     - Gradient Boosted Trees
#     - Support Vector Machines
#     - Many, many more

# ## Part 2: Introduce the Data
# 
# Task: Given attributes about a person, predict whether their income is <=50K or >50K
# Import data and take a look
import numpy as np
import pandas as pd

df = pd.read_csv('adult.csv', na_values=['#NAME?'])
print(df.head(5))
# Take a look at the outcome variable: 'income'
print(df['income'].value_counts())
# Assign outcome as 0 if income <=50K and as 1 if income >50K
df['income'] = [0 if x == '<=50K' else 1 for x in df['income']]

# Assign X as a DataFrame of features and y as a Series of the outcome variable
X = df.drop('income', 1)
y = df.income
print(X.head(5))
print(y.head(5))


# ## Part 3: Basic data cleaning

# ### A. Dealing with data types
# 
# - There are three main data types:
#     - Numeric, e.g. income, age
#     - Categorical, e.g. gender, nationality 
#     - Ordinal, e.g. low/medium/high
#     
#     
# - Models can only handle numeric features
# 
# 
# - Must convert categorical and ordinal features into numeric features
#     - Create dummy features
#     - Transform a categorical feature into a set of dummy features, each representing a unique category
#     - In the set of dummy features, 1 indicates that the observation belongs to that category
# Education is a categorical feature:
print(X['education'].head(5))
# Use get_dummies in pandas 
# Another option: OneHotEncoder in sci-kit learn
print(pd.get_dummies(X['education']).head(5))
# Decide which categorical variables you want to use in model
for col_name in X.columns:
    if X[col_name].dtypes == 'object':
        unique_cat = len(X[col_name].unique())
        print("Feature '{col_name}' has {unique_cat} unique categories".format(col_name=col_name, unique_cat=unique_cat))
# Although, 'native_country' has a lot of unique categories, most categories only have a few observations
print(X['native_country'].value_counts().sort_values(ascending=False).head(10))
# In this case, bucket low frequecy categories as "Other"
X['native_country'] = ['United-States ' if x == 'United-States' else 'Other' for x in X['native_country']]

print(X['native_country'].value_counts().sort_values(ascending=False))

# Create a list of features to dummy
todummy_list = ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country']
# Function to dummy all the categorical variables used for modeling
def dummy_df(df, todummy_list):
    for x in todummy_list:
        dummies = pd.get_dummies(df[x], prefix=x, dummy_na=False)
        df = df.drop(x, 1)
        df = pd.concat([df, dummies], axis=1)
    return df


X = dummy_df(X, todummy_list)
print(X.head(5))


# ### B. Handling missing data
# 
# - Models can not handle missing data
# 
# 
# - Simplest solution
#     - Remove observations/features that have missing data
#     
# 
# - But, removing missing data can introduce a lot of issues
#     - Data is randomly missing: potentially lose a lot of your data
#     - Data is non-randomly missing: in addition to losing data, you are also introducing potential biases
#     - Usually, this is a poor solution
# 
# 
# - An alternative solution is to use imputation
#     - Replace missing value with another value
#     - Strategies: mean, median, highest frequency value of given feature
# How much of your data is missing?
X.isnull().sum().sort_values(ascending=False).head()

# Impute missing values using Imputer in sklearn.preprocessing
from sklearn.preprocessing import Imputer

imp = Imputer(missing_values='NaN', strategy='median', axis=0)
imp.fit(X)
X = pd.DataFrame(data=imp.transform(X) , columns=X.columns)
# Now check again to see if you still have missing data
X.isnull().sum().sort_values(ascending=False).head()


# ## Part 4: More Data Exploration
# 
# - A large portion of the pre-modeling and modeling workflow can be generalized and automated
# 
# 
# - But understanding the problem, domain, and data is extremely important for building high performing models
# 
# 
# - This section covers some tools used for exploring your data to make smarter decisions

# ### A. Outlier detection
# 
# - An outlier is an observation that deviates drastically from other observations in a dataset
# 
# 
# - Occurrence:
#     - Natural, e.g. Mark Zuckerberg's income
#     - Error, e.g. human weight of 2,000 lb. due to mistyping extra 0
# 
# 
# - Why are they problematic?
#     - Naturally occuring:
#         - Not necessarily problematic
#         - But can skew your model by affecting the slope (see image below)
#     - Error 
#         - Indicative of data quality issues
#         - Treat in the same way as a missing value, i.e. use imputation
#    
#    
# - Many, many approaches for detecting outliers; we will discuss two of these:
#     - Tukey IQR
#     - Kernel density estimatation
from IPython.display import Image
Image(filename='outliers.jpg')


# ### Outlier detection - Tukey IQR
# - Identifies extreme values in data
# 
# 
# - Outliers are defined as:
#     - Values below Q1-1.5(Q3-Q1) or above Q3+1.5(Q3-Q1)
#  
#  
# - Standard deviation from the mean is another common method to detect extreme values
#     - But it can be problematic:
#         - Assumes normality 
#         - Sensitive to very extreme values
from IPython.display import Image
Image(filename='tukeyiqr.jpg')
# Imagine taken from: 
# http://datapigtechnologies.com/blog/index.php/highlighting-outliers-in-your-data-with-the-tukey-method/
def find_outliers(x):
    q1 = np.percentile(x, 25)
    q3 = np.percentile(x, 75)
    iqr = q3-q1 
    floor = q1 - 1.5*iqr
    ceiling = q3 + 1.5*iqr
    outlier_indices = list(x.index[(x < floor)|(x > ceiling)])
    outlier_values = list(x[outlier_indices])

    return outlier_indices, outlier_values


column_list = []
outlier_indices, outlier_values = find_outliers(df[column_list])
print(np.sort(outlier_values))


# ### Outlier detection - Kernel Density Estimation
# - Non-parametric way to estimate the probability density function of a given feature
# 
# 
# - Can be advantageous compared to extreme value detection (e.g. Tukey IQR)
#     - Capures outliers in bimodal distributions

# In[25]:


from sklearn.preprocessing import scale
from statsmodels.nonparametric.kde import KDEUnivariate

def find_outliers_kde(x):
    x_scaled = scale(list(map(float, x)))
    kde = KDEUnivariate(x_scaled)
    kde.fit(bw="scott", fft=True)
    pred = kde.evaluate(x_scaled)
    
    n = sum(pred < 0.05)
    outlier_ind = np.asarray(pred).argsort()[:n]
    outlier_value = np.asarray(x)[outlier_ind]

    return outlier_ind, outlier_value


kde_indices, kde_values = find_outliers_kde(X['age'])
print(np.sort(kde_values))

list = []
target_var = 'default'
for i in list:
    print(pd.crosstab(df[target_var],df[i]))
    print(df[[i,target_var]].groupby([i]).mean().sort_values(by= target_var, ascending=False))


# ### B. Distribution of Features
# 
# - A histogram is a simple representation of the distribution of values for a given feature
# 
# 
# - X-axis represents value bins and y-axis represents the frequency of an observation falling into that bin
# 
# 
# - It is also interesting to look at distributions broken up by outcome categories

# Use pyplot in matplotlib to plot histograms
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

def plot_histogram(x):
    plt.hist(x, color='gray', alpha=0.5)
    plt.title("Histogram of '{var_name}'".format(var_name=x.name))
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()

plot_histogram(X['age'])
# Plot histograms to show distribution of features by outcome categories
def plot_histogram_dv(x,y):
    plt.hist(list(x[y==0]), alpha=0.5, label='Outcome=0')
    plt.hist(list(x[y==1]), alpha=0.5, label='Outcome=1')
    plt.title("Histogram of '{var_name}' by Outcome Category".format(var_name=x.name))
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend(loc='upper right')
    plt.show()

plot_histogram_dv(X['age'], y)


# ## Part 5: Feature Engineering

# ### A. Interactions amongst features
# - A simple two-way interaction is represented by: 
#     - X3 = X1 * X2, where X3 is the interaction between X1 and X2
# 
# 
# - Can add interaction terms as additional new features to your model; useful for model if the impact of two or more features on the outcome is non-additive
# 
# 
# - Example: 
#     - Interaction: education and political ideology; outcome: concerns about climate change
#     - While an increase in education amongst liberals or moderates increases concerns about climate change, an increase in education amongst conservatives has the opposite effect
#     - The education-political ideology interaction captures more than the two features alone
# 
# 
# - Note that the interactions amongst dummy variables belonging to the same categorical feature are always zero
# 
# 
# - Although it is very easy to calculate two-way interactions amongst all features, it is very computationally expensive
#     - 10 features = 45 two-way interaction terms
#     - 50 features = 1,225 two-way interaction terms
#     - 100 features = 4,950 two-way interaction terms
#     - 500 features = 124,750 two-way interation terms
#     - Recommend understanding your data and domain if possible and selectively choosing interaction terms
from IPython.display import Image
Image(filename='interactions.jpg')

# Use PolynomialFeatures in sklearn.preprocessing to create two-way interactions for all features
from itertools import combinations
from sklearn.preprocessing import PolynomialFeatures

def add_interactions(df):
    # Get feature names
    combos = list(combinations(list(df.columns), 2))
    colnames = list(df.columns) + ['_'.join(x) for x in combos]
    
    # Find interactions
    poly = PolynomialFeatures(interaction_only=True, include_bias=False)
    df = poly.fit_transform(df)
    df = pd.DataFrame(df)
    df.columns = colnames
    
    # Remove interaction terms with all 0 values            
    noint_indicies = [i for i, x in enumerate(list((df == 0).all())) if x]
    df = df.drop(df.columns[noint_indicies], axis=1)
    
    return df

X = add_interactions(X)
print(X.head(5))


# ### B. Dimensionality reduction using PCA
# - Principal component analysis (PCA) is a technique that transforms a dataset of many features into principal components that "summarize" the variance that underlies the data
# 
# 
# - Each principal component is calculated by finding the linear combination of features that maximizes variance, while also ensuring zero correlation with the previously calculated principal components 
# 
# 
# - Use cases for modeling:
#     - One of the most common dimensionality reduction techniques
#     - Use if there are too many features or if observation/feature ratio is poor
#     - Also, potentially good option if there are a lot of highly correlated variables in your dataset
# 
# 
# - Unfortunately, PCA makes models a lot harder to interpret

from IPython.display import Image
Image(filename='pca.jpg')
# Imagine taken from: 
# http://www.mathworks.com/help/stats/quality-of-life-in-u-s-cities.html?requestedDomain=www.mathworks.com
# Use PCA from sklearn.decompostion to find principal components
from sklearn.decomposition import PCA

pca = PCA(n_components=10)
X_pca = pd.DataFrame(pca.fit_transform(X))
print(X_pca.head(5))


# ## Part 6: Feature selection and model building

# ### Build model using processed data
# Use train_test_split in sklearn.cross_validation to split data into train and test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.70, random_state=1)
# The total number of features have grown substantially after dummying and adding interaction terms
print(df.shape)
print(X.shape)
# Such a large set of features can cause overfitting and also slow computing
# Use feature selection to select the most important features
import sklearn.feature_selection

select = sklearn.feature_selection.SelectKBest(k=20)
selected_features = select.fit(X_train, y_train)
indices_selected = selected_features.get_support(indices=True)
colnames_selected = [X.columns[i] for i in indices_selected]

X_train_selected = X_train[colnames_selected]
X_test_selected = X_test[colnames_selected]
print(colnames_selected)

# Function to build model and find model performance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

def find_model_perf(X_train, y_train, X_test, y_test):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_hat = [x[1] for x in model.predict_proba(X_test)]
    auc = roc_auc_score(y_test, y_hat)
    
    return auc

# Find performance of model using preprocessed data
auc_processed = find_model_perf(X_train_selected, y_train, X_test_selected, y_test)
print(auc_processed)


# ### Build model using unprocessed data

# Drop missing values so model does not throw any error
df_unprocessed = df
df_unprocessed = df_unprocessed.dropna(axis=0, how='any')
print(df.shape)
print(df_unprocessed.shape)
# Remove non-numeric columns so model does not throw an error
for col_name in df_unprocessed.columns:
    if df_unprocessed[col_name].dtypes not in ['int32','int64','float32','float64']:
        df_unprocessed = df_unprocessed.drop(col_name, 1)

# Split into features and outcomes
X_unprocessed = df_unprocessed.drop('income', 1)
y_unprocessed = df_unprocessed.income
# Take a look again at what the unprocessed feature set looks like
print(X_unprocessed.head(5))

# Split unprocessed data into train and test set
# Build model and assess performance
X_train_unprocessed, X_test_unprocessed, y_train, y_test = train_test_split(
    X_unprocessed, y_unprocessed, train_size=0.70, random_state=1)

auc_unprocessed = find_model_perf(X_train_unprocessed, y_train, X_test_unprocessed, y_test)
print(auc_unprocessed)

# Compare model performance
print('AUC of model with data preprocessing: {auc}'.format(auc=auc_processed))
print('AUC of model with data without preprocessing: {auc}'.format(auc=auc_unprocessed))
per_improve = ((auc_processed-auc_unprocessed)/auc_unprocessed)*100
print('Model improvement of preprocessing: {per_improve}%'.format(per_improve = per_improve))
