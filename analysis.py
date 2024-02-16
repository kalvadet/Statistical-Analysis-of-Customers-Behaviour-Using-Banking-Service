#!/usr/bin/env python
# coding: utf-8

# In[88]:


import pandas as pd


# In[89]:


# Read the CSV file (replace 'dataset.csv' with your actual file name)
df = pd.read_csv('C:\\Users\\user\\OneDrive\\Documents\\analysis1\\Main Sample.csv')

# Display the first few rows of the DataFrame
df.head()


# In[90]:


len(df)
df.columns


# # What is the proportion of customers that are still using banking services compared to those that have left in the period covered in the dataset?
# 

# In[73]:


df['IsActiveMember']= df['IsActiveMember'].replace({0:'No',1:'Yes'})
df.head()


# In[74]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[77]:


gender_counts = df['IsActiveMember'].value_counts(normalize=True) * 100


# In[80]:


sns.set_style('whitegrid')
ax = sns.barplot(x=gender_counts.index, y=gender_counts.values, palette='Set2')
for p in ax.patches:
    height = p.get_height()
    ax.annotate(f'{height:.1f}%', (p.get_x() + p.get_width() / 2, height),
                ha='center', va='bottom', fontsize=10)

plt.xlabel('frequency distribution of active and non-active member')
plt.ylabel('Frequency')
plt.title('Distribution of active and non-active member')
plt.show()


# In[19]:


s = pd.Series(df['IsActiveMember'])


# In[30]:


freq_table = s.value_counts().sort_index()


# In[31]:


freq_table


# In[34]:


df = pd.DataFrame({"Is Active Member": freq_table.index, "Frequency": freq_table.values})
c=df['Frequency']/df['Frequency'].sum()


# In[23]:


df


# In[ ]:





# In[91]:


df['HasCreditCard']= df['HasCreditCard'].replace({0:'No',1:'Yes'})
df.head()


# In[92]:


gender_counts = df['HasCreditCard'].value_counts(normalize=True) * 100


# In[93]:


sns.set(style="whitegrid")

sns.set_style('whitegrid')
ax = sns.barplot(x=gender_counts.index, y=gender_counts.values, palette='Set2')
for p in ax.patches:
    height = p.get_height()
    ax.annotate(f'{height:.1f}%', (p.get_x() + p.get_width() / 2, height),
                ha='center', va='bottom', fontsize=10)

plt.xlabel('frequency distribution of credit card status')
plt.ylabel('Frequency')
plt.title('Distribution of credit card status')
plt.show()


# In[51]:


s = pd.Series(df['HasCreditCard'])
freq_table = s.value_counts().sort_index()
freq_table


# In[56]:


f=pd.DataFrame({"HasCreditCard": freq_table.index, "Frequency": freq_table.values})
f['Frequency']/f['Frequency'].sum()


# # What is the relationship between the number of complaints received by the bank authorities and the number of exited customers?
# 

# ### chi square test

# In[57]:


from scipy.stats import chi2_contingency


# In[58]:


contingency_table = pd.crosstab(df['Exited'], df['Complain'])


# In[59]:


chi2_statistic, p_value, _, _ = chi2_contingency(contingency_table)

print(f"Chi-Square Statistic: {chi2_statistic:.4f}")
print(f"P-Value: {p_value:.4f}")


# # What are the characteristics and statistics (in terms of gender, age groups, tenure, etc.) of the customers that are more likely to complain? Provide an informative profile description of those types of customers.
# 

# In[7]:


condition = df[df['Complain']==1]


# In[8]:


condition


# In[9]:


gender_counts = df['Gender'].value_counts(normalize=True) * 100


# In[14]:


sns.set_style('whitegrid')
ax = sns.barplot(x=gender_counts.index, y=gender_counts.values, palette='Set2')
for p in ax.patches:
    height = p.get_height()
    ax.annotate(f'{height:.1f}%', (p.get_x() + p.get_width() / 2, height),
                ha='center', va='bottom', fontsize=10)

plt.xlabel('Gender')
plt.ylabel('Percentage')
plt.title('Distribution of Genders')
plt.show()


# In[15]:


df['Age'].describe()


# In[18]:


sns.set(style='darkgrid')  # Set a style (optional)
sns.histplot(data=df, x='Age', bins=20, kde=True)
plt.xlabel('Median Value of Age for those that complain')
plt.ylabel('Frequency')
plt.title('Distribution of Age')
plt.show()


# In[86]:


df['Tenure'].describe()


# In[ ]:





# In[87]:


sns.set(style='darkgrid')  # Set a style (optional)
sns.histplot(data=df, x='Tenure', bins=20, kde=True)
plt.xlabel('Median Value of tenure for those that complain')
plt.ylabel('Frequency')
plt.title('Distribution of Tenure')
plt.show()


# In[19]:


df['CreditScore'].describe()


# In[20]:


sns.set(style='darkgrid')  # Set a style (optional)
sns.histplot(data=df, x='CreditScore', bins=20, kde=True)
plt.xlabel('Median Value of credit score for those that complain')
plt.ylabel('Frequency')
plt.title('Distribution of credit score')
plt.show()


# In[21]:


df['EstimatedSalary'].describe()


# In[22]:


sns.set(style='darkgrid')  # Set a style (optional)
sns.histplot(data=df, x='EstimatedSalary', bins=20, kde=True)
plt.xlabel('Median Value of Estimated salary for those that complain')
plt.ylabel('Frequency')
plt.title('Distribution of Estimated Salary')
plt.show()


# In[23]:


location_counts = df['Location'].value_counts(normalize=True) * 100


# In[26]:


sns.set_style('whitegrid')
ax = sns.barplot(x=location_counts.index, y=location_counts.values, palette='Set2')
for p in ax.patches:
    height = p.get_height()
    ax.annotate(f'{height:.1f}%', (p.get_x() + p.get_width() / 2, height),
                ha='center', va='bottom', fontsize=10)

plt.xlabel('Location')
plt.ylabel('Percentage')
plt.title('Distribution of Location of those who are likely to complain')
plt.show()


# #  Do the satisfaction scores on complaint resolution provide indication of thecustomers’ likelihood of exiting the bank?
# 

# In[51]:


x = df[['Satisfaction Score']]
y = df['Exited']


# In[41]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[52]:


model = LogisticRegression()



# In[53]:


result=model.fit(x, y)


# In[54]:


print(model.coef_)  # Coefficients
print(model.intercept_)


# In[49]:


import statsmodels.api as sm


# In[55]:


X_train_with_intercept = sm.add_constant(x)
logit_model = sm.Logit(y, X_train_with_intercept)
results = logit_model.fit()

# Print the summary
print(results.summary())


# In[ ]:




