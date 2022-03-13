#!/usr/bin/env python
# coding: utf-8

# ## Import Python Libraries

# In[1]:


import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV

import warnings
warnings.filterwarnings('ignore')


# ## Import Data

# In[2]:


df = pd.read_csv('../input/credit_risk_dataset.csv')
df.head().T


# #### Brief Summary

# In[3]:


df.describe(include='all').T


# #### Get field types - Continuous or Categorical

# In[4]:


df.dtypes


# In[5]:



continuous_columns  = [i for i,j in zip(df.columns,df.dtypes) if j in ['int64','float64']]
categorical_columns = set(df.columns) - set(continuous_columns)
    


# #### Univariate Analysis - Plotting Histograms

# In[6]:


for col in continuous_columns:
    print('Histogram of ', col)
    plt.figure(figsize=(15,3))
    df[col].hist(bins=100)
    plt.show()


# #### Removing Outliers based on Histograms

# In[7]:


df = df.query('person_age<60 and person_income<300000 and person_emp_length<25 and loan_percent_income<0.7')


# In[8]:


for col in continuous_columns:
    print('Histogram of ', col)
    plt.figure(figsize=(15,3))
    df[col].hist(bins=100)
    plt.show()


# #### Find and Impute Missing values

# In[9]:


df.isna().sum()


# - Find values to impute based on custom logic (based on business idea)

# In[10]:


impute_loan_int_rate = df[['loan_intent','loan_grade','loan_int_rate']].groupby(['loan_intent','loan_grade']).mean().reset_index()
impute_loan_int_rate.head(15)


# In[11]:


def find_value_of_loan_int_rate_for_imputation(loan_intent, loan_grade):
    value = impute_loan_int_rate[impute_loan_int_rate['loan_intent']==loan_intent][impute_loan_int_rate['loan_grade']==loan_grade]['loan_int_rate'].values[0]
    return value

find_value_of_loan_int_rate_for_imputation('HOMEIMPROVEMENT', 'A')
    


# In[12]:



print('\nSummary before imputation\n',df['loan_int_rate'].describe())

df['loan_int_rate'] = [find_value_of_loan_int_rate_for_imputation(i, j) if str(k)=='nan' else k for i,j,k in zip(df['loan_intent'], df['loan_grade'],df['loan_int_rate'])]

print('\nSummary after imputation\n',df['loan_int_rate'].describe())


# #### One-Hot Encoding of Categorical Values

# In[13]:


df = pd.get_dummies(data=df, columns=['loan_grade','cb_person_default_on_file','loan_intent','person_home_ownership'], drop_first=True)
df.head().T


# #### Normalize dataframe

# In[14]:


normalized_df = (df-df.min())/(df.max()-df.min())
normalized_df.head().T


# #### Plot Corelation amoung all columns

# In[15]:



plt.figure(figsize=(18,10))
corr_ = normalized_df.corr()
mask = np.triu(np.ones_like(corr_, dtype=bool))
sns.heatmap(corr_, mask=mask, annot=True, fmt='.2f',  cmap="YlGnBu")
plt.show()


# #### Find class distribution

# In[16]:


cls = pd.DataFrame(normalized_df.groupby('loan_status').count()['person_age'])
cls.columns = ['count']
cls['precentage'] = round(100*cls['count']/sum(cls['count']),2)
cls


# #### Prepare dataset for Training and Validation

# In[17]:



X, y = df.drop(columns='loan_status'), df['loan_status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y)


# #### Try Logistic Regression

# In[18]:


lr = LogisticRegression().fit(X_train, y_train)
y_pred = lr.predict(X_test)

print(confusion_matrix(y_test, y_pred),
     '\n',
     classification_report(y_test, y_pred))


# #### Try XGBoost Classifier

# In[19]:


xgb = XGBClassifier().fit(X_train, y_train)
y_pred = xgb.predict(X_test)

print(confusion_matrix(y_test, y_pred),
     '\n',
     classification_report(y_test, y_pred))


# #### Parameter tuning to find better solutions

# In[ ]:


clf = XGBClassifier()

param_grid = {
                'silent'       : [False],
                'max_depth'    : [3,5,8],
                'learning_rate': [ 0.1, 0.2, 0,3],
                'subsample'    : [0.5, 0.75, 1.0],
                'n_estimators' : [100, 500]
             }

rs_clf = RandomizedSearchCV(clf, 
                            param_grid, 
                            n_iter=20,
                            n_jobs=1, 
                            verbose=2, 
                            cv=2,
                            scoring='neg_log_loss', 
                            refit=False, 
                            random_state=42).fit(X_train, y_train)

print('Best parameters = ', rs_clf.best_params)


# #### Check improvement with parameter tuning (if any)

# In[22]:


param = {'subsample': 0.75, 'silent': False, 'n_estimators': 500, 'max_depth': 5, 'learning_rate': 0.1}
                    
xgb = XGBClassifier(param=param).fit(X_train, y_train)
y_pred = xgb.predict(X_test)

print(confusion_matrix(y_test, y_pred),
     '\n',
     classification_report(y_test, y_pred))


# #### Check improvement by tuning threshold

# In[23]:


y_proba = [i[0] for i in xgb.predict_proba(X_test)]

plt.figure(figsize=(15,5))
pd.Series(y_proba).hist(bins=100)
plt.show()


# In[24]:


y_proba = [i[0] for i in xgb.predict_proba(X_test)]

for t in [0.4, 0.45, 0.5, 0.55, 0.6]:
    print('Threshold=', t)
    y_binary = [int(i<t) for i in y_proba]
    
    print(confusion_matrix(y_test, y_binary),
         '\n',
         classification_report(y_test, y_binary))
    
    


# In[25]:



result = pd.DataFrame()
result['actual'] = y_test
result['binary'] = xgb.predict(X_test)
result['score']  = [i[0] for i in xgb.predict_proba(X_test)]

result.head()


# #### Check improvement introducing Manual Reviews

# In[26]:



print('With respect to Binary\n',
      confusion_matrix(result['actual'], result['binary']),
     '\n',
     classification_report(result['actual'], result['binary']))

print('\n\n', '*'*100, '\nFinding best threshold')
for i in [0.3, 0.4, 0.5, 0.6, 0.7]:
    print('Threshold =', i)
    a = result[result['score']<i]
    b = result[result['score']>i+0.2]
    result_select = pd.concat([a,b])
    
#result_select = result.query('score<0.3 or score>0.5')
    print('Manual review = ',len(result)-len(result_select))


    print('With respect to Score\n',
          confusion_matrix(result_select['actual'], result_select['binary']),
         '\n',
         classification_report(result_select['actual'], result_select['binary']))


# In[ ]:




