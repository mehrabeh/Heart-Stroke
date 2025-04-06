#!/usr/bin/env python
# coding: utf-8

# In[362]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from numpy import mean
from numpy import std
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.inspection import permutation_importance
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import shapiro


# In[328]:


df = pd.read_csv(f'D:\machine learning(sharif)\my project/healthcare-dataset-stroke-data.csv')
df.head()


# In[329]:


df.drop('id', axis=1, inplace=True)


# In[330]:


df.head(5)


# ## Data Preprocessing

# In[331]:


df.describe()


# In[332]:


df.info()


# In[333]:


sns.heatmap(df.isnull())


# In[334]:


df.isnull().sum()


# In[335]:


df['bmi'].fillna(df['bmi'].mean(), inplace=True)


# In[336]:


df['bmi'].isnull().sum()


# In[337]:


df.nunique()


# In[338]:


sns.set_style('whitegrid')
sns.pairplot(df)
plt.show()


# In[339]:


plt.figure(figsize = (30, 25))
sns.heatmap(df.corr(), annot = True, cmap="coolwarm")
plt.show()


# In[340]:


sns.set_style('whitegrid')
sns.countplot(x='stroke',hue='gender',data=df,palette='rainbow')


# In[341]:


sns.set_style('whitegrid')
sns.countplot(x='stroke',hue='heart_disease',data=df,palette='rainbow')


# In[342]:


sns.set_style('whitegrid')
sns.countplot(x='stroke',hue='ever_married',data=df,palette='rainbow')


# In[343]:


sns.set_style('whitegrid')
sns.countplot(x='stroke',hue='smoking_status',data=df,palette='rainbow')


# In[344]:


sns.jointplot(y='avg_glucose_level',x='age',data=df,kind='scatter')


# ## Normality Test 

# In[345]:


df['age'].hist(bins=30,color='darkred',alpha=0.7)


# In[346]:


qqplot(df['age'],line='s')


# In[347]:


Statistics, p = shapiro(df['age'])
print(f'Statistics={Statistics:0.3f} p_value={p:0.3f}')

alpha = 0.05

if p >= alpha:
    print('Sample looks Gaussian (fail to reject H0)')
else:
    print('Sample does not look Gaussian (reject H0)')


# In[348]:


df['avg_glucose_level'].hist(bins=30,color='darkred',alpha=0.7)


# In[349]:


qqplot(df['avg_glucose_level'],line='s')


# In[350]:


Statistics, p = shapiro(df['avg_glucose_level'])
print(f'Statistics={Statistics:0.3f} p_value={p:0.3f}')

alpha = 0.05

if p >= alpha:
    print('Sample looks Gaussian (fail to reject H0)')
else:
    print('Sample does not look Gaussian (reject H0)')


# In[351]:


df['bmi'].hist(bins=30,color='darkred',alpha=0.7)


# In[352]:


qqplot(df['bmi'],line='s')


# ## Data Transform

# In[353]:


df['ever_married'].replace({'Yes':1, 'No':0}, inplace=True)
df['gender'].replace({'Male':1, 'Female':0,'Other':2}, inplace=True)
df['Residence_type'].replace({'Urban':1, 'Rural':0}, inplace=True)
df['smoking_status'].replace({'formerly smoked':0, 'never smoked':1, 'smokes':2, 'Unknown':3}, inplace=True)
df['work_type'].replace({'Private':0, 'Self-employed':1, 'children':2, 'Govt_job':3, 'Never_worked':4}, inplace=True)


# In[354]:


df.head()


# In[355]:


X, y = df.drop('stroke', axis=1), df['stroke']
print(X.shape, y.shape)
numerical_ix = X.select_dtypes(include=['int64', 'float64', "float32"]).columns
print("numerical_ix: ",numerical_ix)
categorical_ix = X.select_dtypes(include=['object','bool']).columns
print("categorical_ix: ",categorical_ix)
t = [('cat', OneHotEncoder(), categorical_ix), ('num', MinMaxScaler(), numerical_ix)]
col_transform = ColumnTransformer(transformers=t)


# ## Model Training and Testing

# ### Logistic Regression

# In[356]:


model=LogisticRegression()
cv = RepeatedStratifiedKFold(n_splits=10,n_repeats=5, random_state=1)
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv,n_jobs=-1)
Accuracy_lg=mean(scores)
print('Accuracy_lg: %.3f (%.3f)' % (mean(scores), std(scores)))


# ### SVM

# In[357]:


model=SVC()
pipeline = Pipeline(steps=[('prep',col_transform), ('m', model)])
cv = RepeatedStratifiedKFold(n_splits=10,n_repeats=5, random_state=1)
scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv,n_jobs=-1)
Accuracy_svm=mean(scores)
print('Accuracy_svm: %.3f (%.3f)' % (mean(scores), std(scores)))


# ### K-Nearest Neighbors (KNN)

# In[358]:


model=KNeighborsClassifier()
pipeline = Pipeline(steps=[('prep',col_transform), ('m', model)])
cv = RepeatedStratifiedKFold(n_splits=10,n_repeats=5, random_state=1)
scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv,n_jobs=-1)
Accuracy_knn=mean(scores)
print('Accuracy_knn: %.3f (%.3f)' % (mean(scores), std(scores)))


# ### Decision Tree Classifier

# In[359]:


model=DecisionTreeClassifier()
pipeline = Pipeline(steps=[('prep',col_transform), ('m', model)])
cv = RepeatedStratifiedKFold(n_splits=10,n_repeats=5, random_state=1)
scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv,n_jobs=-1)
Accuracy_dt=mean(scores)
print('Accuracy__dt: %.3f (%.3f)' % (mean(scores), std(scores)))


# ## Model Comparison

# In[360]:


models = ['Logistic Regression', 'SVM', 'KNN','Decision Tree']
accuracy = [Accuracy_lg, Accuracy_svm, Accuracy_knn,Accuracy_dt]
plt.figure(figsize=(10,5))
plt.bar(models, accuracy, color = 'Maroon', width = 0.4)
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.show()


# ### Permutation Feature Importance

# In[361]:


model = KNeighborsClassifier()
model.fit(X, y)
results = permutation_importance(model, X, y, scoring='accuracy')
importance = results.importances_mean
for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))
plt.bar([x for x in range(len(importance))], importance)
plt.show()


# ## Conclusion

# The model accuracies of Logistic Regression and SVM are quite similar 95.1 %. The accuracy of KNN and Decision Tree Classifier are 94.9 % and 91 % So, we can use any of these models to predict the heart stroke.
# 
# The relationship of different features was depicted, but at the end, based on the KNN model, the characteristics that had the greatest impact on the prediction were identified. These features are respectively:
# 1-age
# 2-avg_glucose_level
# 3-bmi

# In[ ]:




