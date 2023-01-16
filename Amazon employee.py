# %%
# import necessary libraries
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 

import warnings
warnings.filterwarnings('ignore')

# %%
# load train data
train = pd.read_csv('amazontrain.csv')

# %%
# load test data
test = pd.read_csv('amazontest.csv')

# %%
# showing first 5 rows of train data
train.head()

# %% [markdown]
# it's a classification problem. The target column is "ACTION".

# %%
# Check the shape of the train data
print('Training shape:',train.shape)

# %%
# showing first 5 rows of test data
test.head()

# %%
# checking the shape of the test data
print('Test shape:',test.shape)

# %% [markdown]
# *Install CatBoost library*

# %%
#!pip install catboost

# %%
# check unique values in each of the columns
train.apply(lambda x: len(x.unique()))

# %% [markdown]
# *Distribution of features*

# %%
# distplot
for i in train.describe().columns:
    sns.distplot(train[i].dropna())
    plt.show()

# %%
# checking the null values
train.isnull().sum().max()

# %% [markdown]
# - good we have no null values.

# %%
# check the value count of target variable
train['ACTION'].value_counts()

# %% [markdown]
# - This data is imbalanced data.most of the values 1(30872), and 0(1897) very low compared to 1.

# %%
print("Allowed", round(train['ACTION'].value_counts()[1]/len(train)*100,2),'% of the dataset.')
print("Denied", round(train['ACTION'].value_counts()[0]/len(train)*100,2),'% of the dataset.')

# %% [markdown]
# we need to take care of the imbalanced dataset later.

# %%
sns.countplot('ACTION', data=train, color='orange')
plt.title("Data Distributions\n (0: Denied || 1: Allowed)", fontsize=14)
plt.show()

# %% [markdown]
# *Resampling*

# %%
# class count
count_class_1, count_class_0 = train['ACTION'].value_counts()

# divide by class
df_class_1 = train[train['ACTION'] == 1]
df_class_0 = train[train['ACTION'] == 0]

# %% [markdown]
# *Random oversampling*

# %%
df_class_0_over = df_class_0.sample(count_class_1, replace=True)
df_test_over = pd.concat([df_class_1, df_class_0_over], axis=0)

print("Random over-sampling")
print(df_test_over['ACTION'].value_counts())

df_test_over.ACTION.value_counts().plot(kind='bar', title='Count(ACTION)', color='orange')
plt.show()

# %%
#boxplot
for i in df_test_over.describe().columns:
    sns.boxplot(df_test_over[i].dropna())
    plt.show()

# %% [markdown]
# the boxplot showing there is lot of outliers in this dataset. we have to more care about that.

# %%
# correlation heatmap
plt.figure(figsize=(20,20))
sns.heatmap(df_test_over.corr(), annot=True, cmap='YlGn', linewidths=0.5, square=True, linecolor='black')
plt.title('Correlation among the variables')
plt.show()

# %% [markdown]
# #### **Model Building**

# %%
X = df_test_over.drop('ACTION', axis=1)
y = df_test_over['ACTION']
X_test = test.drop('id', axis=1)

# %%
df_test_over.shape

# %%
X_train, X_valid, y_train, y_valid = train_test_split(X,y, test_size=.25, random_state=42)

# %% [markdown]
# ### **CatBoost Classifier**

# %%
from catboost import CatBoostClassifier

# %%
%%time

catmodel_1 = CatBoostClassifier(loss_function='Logloss', eval_metric='AUC', verbose=200, random_seed=1) 
catmodel_1.fit(X_train, y_train, eval_set=(X_valid, y_valid), use_best_model = True)

# %%
X.head()

# %%
# specify which features are categorical
categorical_features = list(range(X.shape[1]))
categorical_features

# %%
%%time

catmodel = CatBoostClassifier(loss_function='Logloss', eval_metric='AUC', cat_features=categorical_features, verbose=200, random_seed=1)
catmodel.fit(X_train, y_train, eval_set=(X_valid, y_valid),use_best_model=True)

# %%
# feature importance
feature_imp = catmodel.get_feature_importance(prettified=True)
feature_imp

# %%
predictions = catmodel.predict_proba(X_test)
predictions

# %%
predictions = catmodel.predict(X_test)
predictions

# %%
catmodel.score(X_valid, y_valid)*100

# %% [markdown]
# ## **Logistic Regression**

# %% [markdown]
# using logistic regression algorithms to this same dataset to check which one is working better.

# %%
from sklearn.linear_model import LogisticRegression

# %%
lr = LogisticRegression(max_iter=2500)
lr.fit(X_train, y_train)

# %%
# make predictions
y_pred = lr.predict(X_test)
y_pred

# %%
# measure accuracy
from sklearn.metrics import accuracy_score

print('Train Accuracy:',np.round(accuracy_score(y_train, lr.predict(X_train)), 2)*100)

# %%
lr.score(X_valid, y_valid)*100

# %% [markdown]
# - Compare these 2 algorithms CatBoost giving better accuracy than logistic regression.


