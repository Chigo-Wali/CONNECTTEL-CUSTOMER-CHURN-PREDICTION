#!/usr/bin/env python
# coding: utf-8

# In[2]:


import warnings
warnings.filterwarnings("ignore")


# In[3]:


import pandas as pd # Data processing
import matplotlib.pyplot as plt # Data visualization
import seaborn as sns # Data visualization
from collections import Counter # Counting
import missingno as msno # Missing Data visualization

pd.set_option('display.max.rows',130)
pd.set_option('display.max.columns',130)
pd.set_option('float_format', '{:.2f}'.format)


# In[4]:


pip install --upgrade imbalanced-learn


# In[5]:


from imblearn.over_sampling import SMOTE


# In[6]:


# Import libraries
import pandas as pd # Data analysis
import numpy as np # Data analysis
import matplotlib.pyplot as plt # Data visualization
import seaborn as sns # Data visualization
from collections import Counter # Counting
import missingno as msno # Missing Data visualization

# Data Pre-processing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Classifier Libraries, which are Machine Learning algorithms
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Machine Learning and Evaluation
from sklearn.cluster import KMeans
#from sklearn.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score, homogeneity_score

# Evaluaton metrics

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

import warnings
warnings.filterwarnings("ignore")


# In[7]:


get_ipython().run_line_magic('matplotlib', 'inline')


# ## Importing dataset

# In[8]:


connecttel_df = pd.read_csv(r"/Users/WaliNelson/Downloads/Customer-Churn.csv")


# In[7]:


connecttel_df.head()


# In[8]:


#  Dataset Info

connecttel_df.info()


# In[11]:


# Converting the column named SeniorCitizen from integer to object

connecttel_df['SeniorCitizen'] = connecttel_df['SeniorCitizen'].astype(str)
connecttel_df['SeniorCitizen'] = connecttel_df['SeniorCitizen'].replace({'0': 'No', '1': 'Yes'})


# In[9]:


# Converting the column named tenure and TotalCharges from object to integer and float

connecttel_df['tenure'] = connecttel_df['tenure'].astype(int)

# connecttel_df['TotalCharges'] = connecttel_df['TotalCharges'].astype(float)

connecttel_df['TotalCharges'] = pd.to_numeric(connecttel_df['TotalCharges'], errors='coerce').fillna(0) # # If there are non-numeric values, convert them to 0


# In[13]:


connecttel_df.info(verbose = True)


# In[15]:


# Shape of the Dataset

connecttel_df.shape


# There are 2 numerical variables

# ### Checking for missing values

# In[12]:


connecttel_df.isnull().sum()


# ### Checking for duplicate values

# In[13]:


connecttel_df.duplicated().sum()


# In[14]:


# Visualizing the missing values

sns.heatmap(connecttel_df.isnull())


# ## What is discovered about the dataset
# 
# The dataset is from ConnectTel. The dataset consists of 7043 rows and 21 columns of customer activity data. There arer neither missing nor duplicate values in the dataset.

# ### Identifying the columns that are categorical variables

# In[15]:


connecttel_df.select_dtypes(include = 'object').columns


# In[16]:


len(connecttel_df.select_dtypes(include = 'object').columns)


# There are 19 categorical variables

# ### Identifying the columns that are numerical variables

# In[17]:


connecttel_df.select_dtypes(include = ['int64', 'float64']).columns


# In[18]:


# Description of the dataset showing the numerical data

connecttel_df.describe()


# # Variable Description
# 
# - CustomerID: A unique identifier assigned to each telecom customer, enabling tracking and identification of individual customers.
# 
# - Gender: The gender of the customer, which can be categorized as male, or female. This information helps in analyzing gender-based trends in customer churn.
# 
# - SeniorCitizen: A binary indicator that identifies whether the customer is a senior citizen or not. This attribute helps in understanding if there are any specific churn patterns among senior customers.
# 
# - Partner: Indicates whether the customer has a partner or not. This attribute helps in evaluating the impact of having a partner on churn behavior.
# - Dependents: Indicates whether the customer has dependents or not. This attribute helps in assessing the influence of having dependents on customer churn.
# 
# - Tenure: The duration for which the customer has been subscribed to the telecom service. It represents the loyalty or longevity of the customer’s relationship with the company and is a significant predictor of churn.
# - PhoneService: Indicates whether the customer has a phone service or not. This attribute helps in understanding the impact of phone service on churn.
# 
# - MultipleLines: Indicates whether the customer has multiple lines or not. This attribute helps in analyzing the effect of having multiple lines on customer churn.
# 
# - InternetService: Indicates the type of internet service subscribed by the customer, such as DSL, fiber optic, or no internet service. It helps in evaluating the relationship between internet service and churn.
# 
# - OnlineSecurity: Indicates whether the customer has online security services or not. This attribute helps in analyzing the impact of online security on customer churn.
# 
# - OnlineBackup: Indicates whether the customer has online backup services or not. This attribute helps in evaluating the impact of online backup on churn behavior.
# 
# - DeviceProtection: Indicates whether the customer has device protection services or not. This attribute helps in understanding the influence of device protection on churn.
# 
# - TechSupport: Indicates whether the customer has technical support services or not. This attribute helps in assessing the impact of tech support on churn behavior.
# 
# - StreamingTV: Indicates whether the customer has streaming TV services or not. This attribute helps in evaluating the impact of streaming TV on customer churn.
# 
# - StreamingMovies: Indicates whether the customer has streaming movie services or not. This attribute helps in understanding the influence of streaming movies on churn behavior.
# 
# - Contract: Indicates the type of contract the customer has, such as a month-to-month, one-year, or two-year contract. It is a crucial factor in predicting churn as different contract lengths may have varying impacts on customer loyalty.
# 
# - PaperlessBilling: Indicates whether the customer has opted for paperless billing or not. This attribute helps in analyzing the effect of paperless billing on customer churn.
# 
# - PaymentMethod: Indicates the method of payment used by the customer, such as electronic checks, mailed checks, bank transfers, or credit cards. This attribute helps in evaluating the impact of payment methods on churn.
# 
# - MonthlyCharges: The amount charged to the customer on a monthly basis. It helps in understanding the relationship between monthly charges and churn behavior.
# 
# - TotalCharges: The total amount charged to the customer over the entire tenure. It represents the cumulative revenue generated from the customer and may have an impact on churn.
# 
# - Churn: The target variable indicates whether the customer has churned (canceled the service) or not. It is the main variable to predict in telecom customer churn analysis.

# In[19]:


# Check unique values for each variable

for c in connecttel_df.columns:
    print("Number of unique values in ",c,"is",connecttel_df[c].nunique())


# In[21]:


# % of total customers churning

perc_churn = (churn_df.Churn.count()/connecttel_df.Churn.count())*100
print(f"Percentage of customer churning : {round(perc_churn,2)}%")


# In[17]:


senior_count = connecttel_df['SeniorCitizen'].eq('1').sum()
print("The sum of Senior Citizens in the dataset is:", senior_count, "out of 7043 customers")

senior_df = connecttel_df[connecttel_df['SeniorCitizen'].eq('1')]

senior_df


# In[23]:


seniorchurn_count = senior_df['Churn'].eq('Yes').sum()
print("The sum of customer churn of Senior Sitizens in the dataset is:", seniorchurn_count, "out of 1142 Senior Citizens")

seniorchurn_df = senior_df[senior_df['Churn'].eq('Yes')]

seniorchurn_df


# In[24]:


# Senior Citizen Churn data

senior_churn = churn_df.groupby(['SeniorCitizen'])['Churn'].value_counts().reset_index(name='SeniorCitizen_churn')
print(senior_churn.sum())
senior_churn


# In[25]:


connecttel_df.head()


# # Univariate Analysis for Categorical Variables

# def countplot_function(dataframe, column, figsize = (15,10), palette = "viridis"):
#     plt.figure(figsize = figsize)
#     sns.countplot(dataframe[column], palette = palatte)
#     plt.title("{} countplot".format(column), fontsize = 15)
#     plt.xlabel("{}".format(column), fontsize = 10)

# countplot_function(dataframe = connecttel_df, column = 'gender', palette = random.choice(palette_values))
# #plt.savefig('Plots/Gender_countplot.png')
# plt.show

# In[123]:


custom_palette = ["#495c78", "#abb5c4"]
ax = sns.countplot(x=connecttel_df["gender"], order=connecttel_df["gender"].value_counts(ascending=False).index, palette=custom_palette)
values = connecttel_df["gender"].value_counts(ascending=False).values
ax.bar_label(container=ax.containers[0], label=values)
plt.title("Total Distribution of Customers by Customer Gender");


# In[124]:


# Total distribution by customer gender

fig,ax = plt.subplots(figsize=(5,5))
count = Counter(connecttel_df["gender"])
ax.pie(count.values(), labels=count.keys(), autopct=lambda p:f'{p:.2f}%')
ax.set_title("Percentage of customers by Gender Pie Chart")
plt.show();


# ## Observation:
# The distribution between the male and female gender are equal, therefore they are represented equally

# In[18]:


# Total distribution by SeniorCitizen

custom_palette = ["#495c78", "#abb5c4"]
ax = sns.countplot(x=connecttel_df["SeniorCitizen"], order=connecttel_df["SeniorCitizen"].value_counts(ascending=False).index, palette=custom_palette)
values = connecttel_df["SeniorCitizen"].value_counts(ascending=False).values
ax.bar_label(container=ax.containers[0], label=values)
plt.title("Total Distribution of Customers by Senior-Citizenship");


# ## Observation
# - Our of a total of 7043 customers 1142 are Senior Citizens which is 16.21% of the total number of customers.
# - Due to the disparity between the Senior Citizens and the rest of the customers, we should be able to predict their behavior and determine the possiblity of churning 

# In[29]:


# Total distribution by Partner

custom_palette = ["#495c78", "#abb5c4"] 
ax = sns.countplot(x=connecttel_df["Partner"], order=connecttel_df["Partner"].value_counts(ascending=False).index, palette=custom_palette)

values = connecttel_df["Partner"].value_counts(ascending=False).values
ax.bar_label(container=ax.containers[0], label=values)
plt.title("Total Distribution of Customers by Partner")

plt.show()


# ## Observation
# - There are more customers who do not have partners than there are who have partners.
# - Since we have more data for those without partner, we should be able to predict their behavior and determine the possiblity of churning 

# In[30]:


# Total distribution by Dependents

custom_palette = ["#495c78", "#abb5c4"] 
ax = sns.countplot(x=connecttel_df["Dependents"], order=connecttel_df["Dependents"].value_counts(ascending=False).index, palette=custom_palette)

values = connecttel_df["Dependents"].value_counts(ascending=False).values
ax.bar_label(container=ax.containers[0], label=values)
plt.title("Total Distribution of Customers by Dependents")

plt.show()


# ## Observation
# - It appears the number of customers that are not dependents is higher than the number of customers that are dependents. This suggests that customers with no dependents constitute a larger segment within the data set.
# - Number of non dependents are 4933 (70.04%), while the Dependents are 2110 (29.96%).

# In[32]:


# Total distribution by PhoneService

custom_palette = ["#495c78", "#abb5c4"] 
ax = sns.countplot(x=connecttel_df["PhoneService"], order=connecttel_df["PhoneService"].value_counts(ascending=False).index, palette=custom_palette)

values = connecttel_df["PhoneService"].value_counts(ascending=False).values
ax.bar_label(container=ax.containers[0], label=values)
plt.title("Total Distribution of Customers by PhoneService")

plt.show()


# In[ ]:





# ## Observation
# - The chart shows that a significantly larger number of customers have phone service ("Yes") compared to those who do not ("No"). The exact proportion is 90.32% with phone service and 9.68% without phone service.
# - It would be interesting to know if there is any relationship between phone service and customer churn. For example, do customers who have phone service tend to be more satisfied with the company and less likely to churn? Analyzing customer data could provide insights into this relationship.

# In[182]:


# Total distribution by Multiple Lines

custom_palette = ["#495c78", "#abb5c4", "#53d4ad"] 
ax = sns.countplot(x=connecttel_df["MultipleLines"], order=connecttel_df["MultipleLines"].value_counts(ascending=False).index, palette=custom_palette)

values = connecttel_df["MultipleLines"].value_counts(ascending=False).values
ax.bar_label(container=ax.containers[0], label=values)
plt.title("Total Distribution of Customers by Multiple Lines")

plt.show()


# ## Observation
# - The majority of customers fall into the "No Multiple Lines" category, represented by the tallest blue bar, which is 48.13% of the customers.
# - There is a smaller but significant number of customers with "Multiple Lines", shown by the shorter bright blue bar, which is 42.18%.
# - The category with the fewest customers is "No Phone Service", indicated by the smallest green bar, which is 9.68%.
# 

# In[184]:


# Total distribution by Internet Service

custom_palette = ["#495c78", "#abb5c4", "#53d4ad"] 
ax = sns.countplot(x=connecttel_df["InternetService"], order=connecttel_df["InternetService"].value_counts(ascending=False).index, palette=custom_palette)

values = connecttel_df["InternetService"].value_counts(ascending=False).values
ax.bar_label(container=ax.containers[0], label=values)
plt.title("Total Distribution of Customers by Internet Service")

plt.show()


# ## Observation
# - From the count on the y-axis, it seems the majority of customers (43.96%) have Fibre Optic internet service. This bar is likely the tallest, indicating its dominance among the options.
# - DSL internet has a smaller but significant number of customers (34.37%), represented by a shorter bar compared to Fibre Optic.
# - Finally, the bar for No Internet appears to be the smallest (21.67%), showcasing the smallest group of customers without any internet service from the two providers.
# 

# In[ ]:





# In[187]:


# Total distribution by Online Security

custom_palette = ["#495c78", "#abb5c4", "#53d4ad"] 
ax = sns.countplot(x=connecttel_df["OnlineSecurity"], order=connecttel_df["OnlineSecurity"].value_counts(ascending=False).index, palette=custom_palette)

values = connecttel_df["OnlineSecurity"].value_counts(ascending=False).values
ax.bar_label(container=ax.containers[0], label=values)
plt.title("Total Distribution of Customers by Online Security")

plt.show()


# ## Observation
# - The majority of customers (49.67%), represented by the tallest blue bar, do not use online security ("No").
# - A smaller but significant number of customers (28.67%) use online security ("Yes"), shown by the shorter bar.
# - The smallest group of customers (21.67%) do not have internet service at all ("No internet service"), indicated by the shortest bar on the right.

# In[ ]:





# In[189]:


# Total distribution by Online Backup

custom_palette = ["#495c78", "#abb5c4", "#53d4ad"] 
ax = sns.countplot(x=connecttel_df["OnlineBackup"], order=connecttel_df["OnlineBackup"].value_counts(ascending=False).index, palette=custom_palette)

values = connecttel_df["OnlineBackup"].value_counts(ascending=False).values
ax.bar_label(container=ax.containers[0], label=values)
plt.title("Total Distribution of Customers by Online Backup")

plt.show()


# ## Observation
# - The bar labeled "No" is the tallest, indicating that the majority of customers (3088, 43.84%) do not use online backup service.
# - The bar labeled "Yes" is shorter but still represents a significant number of customers (2429, 34.49%) who use online backup service.
# - The smallest bar, labeled "No internet service," shows the fewest customers (1526, 21.67%) who lack internet altogether.

# In[192]:


# Total distribution by Device Protection

custom_palette = ["#495c78", "#abb5c4", "#53d4ad"] 
ax = sns.countplot(x=connecttel_df["DeviceProtection"], order=connecttel_df["DeviceProtection"].value_counts(ascending=False).index, palette=custom_palette)

values = connecttel_df["DeviceProtection"].value_counts(ascending=False).values
ax.bar_label(container=ax.containers[0], label=values)
plt.title("Total Distribution of Customers by Device Protection")

plt.show()


# ## Observation
# - The bar labeled "No" is the tallest, indicating that the majority of customers (3095, 43.94%) do not use Device Protection service.
# - The bar labeled "Yes" is shorter but still represents a significant number of customers (2422, 34.39%) who use Device Protection service.
# - The smallest bar, labeled "No internet service," shows the fewest customers (1526, 21.67%) who lack internet altogether.

# In[193]:


# Total distribution by Tech Support
custom_palette = ["#495c78", "#abb5c4", "#53d4ad"] 
ax = sns.countplot(x=connecttel_df["TechSupport"], order=connecttel_df["TechSupport"].value_counts(ascending=False).index, palette=custom_palette)

values = connecttel_df["TechSupport"].value_counts(ascending=False).values
ax.bar_label(container=ax.containers[0], label=values)
plt.title("Total Distribution of Customers by Tech Support")

plt.show()


# ## Observation
# - The bar labeled "No" is the tallest, indicating that the majority of customers (3473, 49.31%) do not use Tech Support.
# - The bar labeled "Yes" is shorter but still represents a significant number of customers (2044, 29.02%) who use Tech Support.
# - The smallest bar, labeled "No internet service," shows the fewest customers (1526, 21.67%) who lack internet altogether.

# In[196]:


# Total distribution by Tech Streaming TV
custom_palette = ["#495c78", "#abb5c4", "#53d4ad"] 
ax = sns.countplot(x=connecttel_df["StreamingTV"], order=connecttel_df["StreamingTV"].value_counts(ascending=False).index, palette=custom_palette)

values = connecttel_df["StreamingTV"].value_counts(ascending=False).values
ax.bar_label(container=ax.containers[0], label=values)
plt.title("Total Distribution of Customers by Streaming TV")

plt.show()


# ## Observation
# - Majority Uses Streaming TV: The bar labeled "Yes" is the tallest, indicating that the majority of customers (2810, 39.9%) use Streaming TV service.
# - Significant Minority Doesn't Use Streaming TV: The bar labeled "No" is shorter but still represents a significant number of customers (2707, 38.44%) who don't use Streaming TV service.
# - Few Customers No Internet: The smallest bar, labeled "No internet service," shows the fewest customers (1526, 21.67%) who lack internet altogether.

# In[199]:


# Total distribution by Tech Streaming Movies
custom_palette = ["#495c78", "#abb5c4", "#53d4ad"] 
ax = sns.countplot(x=connecttel_df["StreamingMovies"], order=connecttel_df["StreamingMovies"].value_counts(ascending=False).index, palette=custom_palette)

values = connecttel_df["StreamingMovies"].value_counts(ascending=False).values
ax.bar_label(container=ax.containers[0], label=values)
plt.title("Total Distribution of Customers by Streaming Movies")

plt.show()


# ## Observation
# - Majority Uses Streaming Movies: The bar labeled "Yes" is the tallest, indicating that the majority of customers (2785, 39.54%) use Streaming Movies service.
# - Minority Doesn't Use Streaming Movies: The bar labeled "No" is shorter but still represents a number of customers (2732, 39.79%) who don't use Streaming Movies service.
# - Few Customers No Internet: The smallest bar, labeled "No internet service," shows the fewest customers (1526, 21.67%) who lack internet altogether.

# In[201]:


# Total distribution by Tech Contract
custom_palette = ["#495c78", "#abb5c4", "#53d4ad"] 
ax = sns.countplot(x=connecttel_df["Contract"], order=connecttel_df["Contract"].value_counts(ascending=False).index, palette=custom_palette)

values = connecttel_df["Contract"].value_counts(ascending=False).values
ax.bar_label(container=ax.containers[0], label=values)
plt.title("Total Distribution of Customers by Contract")

plt.show()


# ## Observation
# - Majority Uses Month-to-Month Contract: The bar labeled "Month-to-Month" is the tallest, indicating that the majority of customers (3875, 55.02%) subscribe to a Month-to-Month Contract plan.
# - Significant Minority Uses Two Year Contract: The bar labeled "Two Year" is significantly shorter and represents a number of customers (1695, 24.07%) who subscribe to a Two Year Contract.
# - Fewer Customers One Year Contract: The bar labeled "One Year" is the shortest and represents a number of customers (1473, 20.91%) who subscribe to a One Year Contract.

# In[42]:


# Total distribution by Paperless Billing
custom_palette = ["#495c78", "#abb5c4", "#4057a3"] 
ax = sns.countplot(x=connecttel_df["PaperlessBilling"], order=connecttel_df["PaperlessBilling"].value_counts(ascending=False).index, palette=custom_palette)

values = connecttel_df["PaperlessBilling"].value_counts(ascending=False).values
ax.bar_label(container=ax.containers[0], label=values)
plt.title("Total Distribution of Customers by Paperless Billing")

plt.show()


# ## Observation
# - The chart shows that a larger number of customers have Paperless Billing ("Yes") compared to those who do not ("No"). The exact proportion is 59.22% with Paperless Billing and 40.78% without Paperless Billing.

# In[203]:


# Total distribution by Payment Method
custom_palette = ["#495c78", "#0e41e8", "#abb5c4", "#53d4ad"] # 53d4ad  0e41e8 4057a3 abb5c4
plt.figure(figsize=(15, 7))
ax = sns.countplot(x=connecttel_df["PaymentMethod"], order=connecttel_df["PaymentMethod"].value_counts(ascending=False).index, palette=custom_palette)

values = connecttel_df["PaymentMethod"].value_counts(ascending=False).values
ax.bar_label(container=ax.containers[0], label=values)
plt.title("Total Distribution of Customers by Payment Method")

plt.show()


# ## Observation
# - Electronic Check Dominates: The bar for Electronic Check is the tallest, indicating it's the most preferred payment method, with 2365 (33.58%) customers using it.
# - Mailed Check in Second: Mailed Check is the second most popular option, with 1612 (22.89%) customers using it.
# - Bank Transfer and Credit Card (Automatic) Less Common: Bank Transfer (Automatic) is used by 1544 (21.92%) customers, while Credit Card (Automatic) is the least common, with only 1522 (21.61%) customers using it.

# In[44]:


# Total distribution by Churn
custom_palette = ["#495c78", "#abb5c4", "#4057a3"] 
ax = sns.countplot(x=connecttel_df["Churn"], order=connecttel_df["Churn"].value_counts(ascending=False).index, palette=custom_palette)

values = connecttel_df["Churn"].value_counts(ascending=False).values
ax.bar_label(container=ax.containers[0], label=values)
plt.title("Total Distribution of Customers by Churn")

plt.show()


# In[65]:


# Total distribution by churn

fig,ax = plt.subplots(figsize=(5,5))
count = Counter(connecttel_df["Churn"])
ax.pie(count.values(), labels=count.keys(), autopct=lambda p:f'{p:.2f}%')
ax.set_title("Percentage of customers by Churn")
plt.show();


# ## Observation
# - We can observe that a larger number and percentage of the customers have not churned.
# - Out of 7043 customers, 5174 (73.46%) remain, while 1869 (26.54%) have churned

# # Univariate Analysis for Numerical Variables

# In[171]:


# Total distribution by Tenure

sns.distplot(connecttel_df['tenure'])
axs[1,1].set_title("Histogram on Tenure")
plt.show;


# ## Observation
# - The distribution appears to be right-skewed, meaning there are more customers with shorter tenures than customers with longer tenures. This is a common finding in customer data, as it's generally easier for customers to churn in the early stages of their relationship with a company.
# - The median tenure might be around 12 months, based on the highest point on the curve. This implies half the customers have tenures less than 12 months and half have more.
# - he distribution seems somewhat spread out, with a tail extending to the right suggesting some customers have significantly longer tenures.

# In[60]:


Monthly_Charges = connecttel_df['MonthlyCharges']
Total_Charges = connecttel_df['TotalCharges']

print(Monthly_Charges.describe())
print(Total_Charges.describe())


# In[48]:


# Histogram of Total distribution by Monthly Charges
plt.figure(figsize=(10, 6))
plt.hist(Monthly_Charges, bins=20, color='skyblue', edgecolor='black')
plt.title('Histogram of Monthly Charges')
plt.show()


# In[59]:


# Converting the Total Charges column from object to float, or numerical

# connecttel_df['TotalCharges'] = connecttel_df['TotalCharges'].astype(float)

connecttel_df['TotalCharges'] = pd.to_numeric(connecttel_df['TotalCharges'], errors='coerce') # # If there are non-numeric values, convert them to NaN


# In[ ]:





# In[61]:


# Kernel Density Plot
plt.figure(figsize=(10, 6))
sns.kdeplot(Total_Charges, fill=True, color='orange')
plt.title('Kernel Density Plot of Total Charges')

plt.show()


# In[62]:


sns.boxplot(x=Total_Charges)
plt.title('Box Plot of Total Charges')
plt.show()


# In[ ]:





# # Bivariate Alalysis - Possible Comparisons 
# 
# ##### Target Variable is Churn: The key variable indicating whether a customer has churned or not.
# 
# ##### Demographic Features:
# - Gender: Examining churn rates across genders could reveal any gender-specific patterns.
# - SeniorCitizen: Comparing churn rates between senior citizens and younger customers could highlight potential differences in retention strategies.
# - Partner: Investigating whether having a partner influences churn behavior.
# - Dependents: Analyzing the impact of having dependents on churn.
# 
# ##### Service Usage Features:
# 
# -    PhoneService: Understanding if having phone service impacts churn.
# -    MultipleLines: Examining whether multiple lines are associated with higher or lower churn rates.
# -    InternetService: Analyzing the effect of different internet service types (DSL, Fiber optic, or none) on churn.
# -    OnlineSecurity: Exploring the relationship between online security subscription and churn.
# -    OnlineBackup: Determining if online backup service usage affects churn.
# -    DeviceProtection: Assessing the impact of device protection on churn.
# -    TechSupport: Examining the association between tech support usage and churn.
# -    StreamingTV: Investigating the link between streaming TV subscription and churn.
# -    StreamingMovies: Assessing the impact of streaming movies on churn.
# 
# ##### Contractual and Billing Features:
# 
# -    Contract: Comparing churn rates across different contract types (month-to-month, one-year, two-year).
# -    PaperlessBilling: Analyzing whether paperless billing preferences influence churn.
# -    PaymentMethod: Exploring if churn rates vary based on payment methods (electronic check, mailed check, bank transfer, credit card).
# - Tenure: Analyzing the relationship between tenure and churn can provide valuable insights into customer loyalty and potential risk factors for losing customers.
# ##### Financial Features:
# 
# -    MonthlyCharges: Assessing the relationship between monthly charges and churn.
# -    TotalCharges: Examining the effect of cumulative charges on churn.
# 
# 
# ##### Target Variable is Tenure:
# 
#  - Tenure + Service Usage: Analyze churn rates for different tenure segments across various service usage patterns (e.g., phone service, internet service, streaming services).
# -    Tenure + Contract: Compare churn rates for different contract types within different tenure segments.
# -    Tenure + Financial Features: Explore the interaction between tenure, monthly charges, and total charges and their impact on churn.

# In[24]:


# Barplot Function

def barplot_function(dataframe, x_value, y_value, title_size=15, label_size=10, figsize=(15, 10), palette="viridis"):
    plt.figure(figsize=figsize)
    sns.barplot(x=x_value, y=y_value, data=dataframe, palette=palette)
    plt.xlabel("{} Value".format(x_value), fontsize=label_size)
    plt.ylabel("{} Value".format(y_value), fontsize=label_size)
    plt.title("{} Vs. {} Barplot".format(x_value, y_value), fontsize=title_size)
    plt.show()

# Example usage:
# barplot_function(your_dataframe, 'x_column', 'y_column')


# In[130]:


# Boxplot Function
def boxplot_function(dataframe, x_value, y_value, title_size = 15, label_size = 10, figsize = (15, 10), palette = "viridis"):
    plt.figure(figsize = (figsize))
    sns.boxplot(x = x_value, y = y_value, data = dataframe, palette = palette)
    plt.xlabel("{} Value".format(x_value), fontsize = label_size)
    plt.ylabel("{} Value".format(y_value), fontsize = label_size)
    plt.title("{} Vs. {} Boxplot".format(x_value, y_value), fontsize = title_size)


# In[30]:


boxplot_function(dataframe = connecttel_df, x_value = "tenure", y_value = "Churn", palette = "YlGnBu")
plt.show()


# ## Observation
# 
# - Tenure distribution: The tenure distribution for churned appears right-skewed with shorter tenures being more common and longer tenures less fequent, while non-churned customers appears to be is a norman distributiom.
# 
# - Median tenure: The median tenure for non-churned customers (around 38 months) is higher than the median tenure for churned customers (around 10 months). This suggests that customers who churn tend to do so earlier in their relationships with the telecommunications company.
# - Box sizes and IQR: The boxes for non-churned customers are wider than the boxes for churned customers, indicating a greater spread in tenures among those who haven't churned. Additionally, the interquartile range (IQR) for non-churned customers is larger, further highlighting the wider range of tenures in this group.
# - There are a few outliers with longer tenures in both the non-churned group. These outliers could represent customers with specific reasons for churning beyond the typical patterns.

# In[125]:


# Sort the DataFrame by 'Tenure' for a chronological order
connecttel_df.sort_values(by='tenure', inplace=True)

# Create a line chart to show the trend of customer churn
plt.figure(figsize=(10, 6))
sns.lineplot(x='tenure', y='Churn', data=connecttel_df, marker='o', color='b')

# Customize the plot
plt.title('Customer Churn Trend Based on Tenure')
plt.xlabel('Tenure')
plt.ylabel('Churn Rate')
plt.grid(True)
plt.show()


# ## Observation
# Overall Trend:
# 
#     The churn rate starts high around 10% for customers with short tenure (around 0-3 months).
#     It then decreases rapidly to around 2% within the first year and continues to gradually decline further as tenure increases.
#     This suggests that a significant portion of customer churn happens early on, and customers who stay beyond the initial period are more likely to remain with the service.
# 
# Key Observations:
# 
#     There's a steep drop in churn rate around the 3-month mark, indicating that this might be a critical period for customer retention efforts.
#     The churn rate stabilizes after around 2 years, suggesting that customers reaching this tenure are highly likely to stay.
# 
# Possible Explanations:
# 
#     The high initial churn rate could be due to various factors like dissatisfaction with the service, difficulty during onboarding, or unmet expectations.
#     The rapid decrease in churn within the first year could be attributed to successful onboarding experiences, resolution of initial issues, or building of trust and value.
#     The gradual decline beyond one year might be influenced by factors like habit formation, switching costs, and increased perceived value of the service with longer use.
# 
# Recommendations:
# 
#     Focus retention efforts on acquiring and onboarding customers effectively to address the initial high churn rate.
#     Analyze reasons for churn within the first few months and implement improvements to the onboarding process or address common pain points.
#     Utilize targeted campaigns and incentives to engage and retain customers during the first year, when they are still forming their loyalty.
#     Consider offering loyalty programs or special perks for long-tenured customers to further incentivize them to stay.

# In[31]:


boxplot_function(dataframe = connecttel_df, x_value = "MonthlyCharges", y_value = "Churn", palette = "YlGnBu")
#plt.savefig('Plots/Churn Vs. MonthlyCharges.png')
plt.show()


# ## Observation
# -    Charge distribution: The distribution of monthly charges appears left-skewed for both churned and non-churned customers, with most customers having higher charges.
# 
# - Median charges: The median monthly charge for churned customers  (around $80) is slightly higher that the one for non-churned cusomers (around $65). This suggests that a simple comparison of medians reveals a strong relationship between higher charges and churn.
# 
# - Box sizes and IQR: The boxes for non-churned customers are wider than the boxes for churned customers, indicating a greater spread in monthly charges among those who haven't churned. Additionally, the interquartile range (IQR) for non-churned customers is slightly larger, further highlighting the wider range of charges in this group.

# In[35]:


print(connecttel_df['MonthlyCharges'].max())
print(connecttel_df['MonthlyCharges'].min())
print(connecttel_df['TotalCharges'].max())
print(connecttel_df['TotalCharges'].min())


# In[37]:


nochurn_count = connecttel_df['Churn'].eq('No').sum()
print("The sum of customer who have not churned in the dataset is:", nochurn_count, "out of 7043 customers")

nochurn_df = connecttel_df[connecttel_df['Churn'].eq('No')]

nochurn_df


# In[38]:


print(churn_df['TotalCharges'].median())
print(nochurn_df['TotalCharges'].median())


# In[29]:


boxplot_function(dataframe = connecttel_df, x_value = "Churn", y_value = "TotalCharges", palette = "YlGnBu")
plt.show()


# ## Observation
#  - Right-skewed distribution: The total charges distribution remains right-skewed for both churned and non-churned customers.
# - Median for Non-Churned Higher: The median total charge for non-churned customers (1683.6) is significantly higher than the median for churned customers (703.55). This notable difference suggests a potential relationship between lower total charges and higher churn risk.
# - Wider Spread for Non-Churned: The boxes and IQR for non-churned customers still indicate a greater spread in total charges among those who haven't churned.
# 
# - High-Charge Outliers Only for Churned: Outliers with higher total charges are present only in the churned group, suggesting specific customer segments with high spending who do not remain loyal.
# 
# - Lower Charges Associated with Churn: The higher median total charge for non-churned customers reinforces the potential association between lower total charges and increased churn risk.
#  - Cumulative Effect Still Relevant: The potential for a cumulative effect of charges over time remains a consideration, as total charges represent accumulated spending.
# - Segmentation for Retention: The difference in median charges and the presence of high-charge outliers among churned customers further supports the idea of segmenting customers based on total charges for targeted retention strategies.
# - Further Analysis Crucial: As before, additional analysis is essential to confirm and quantify the relationship between total charges and churn, as well as to explore interactions with other factors.
# 
# - Additional Insights: Potential Loyalty among High-Charge Customers: The presence of outliers with high total charges only among churned customers suggests that customers who spend more with the company might be more likely to remain loyal. Understanding the characteristics and motivations of these high-charge, loyal customers could provide valuable insights for retention strategies.

# In[ ]:


procat = df.groupby("Product_Category")[["total_cost", "total_revenue", "profit"]].sum().reset_index()
# To transpose, or to transform total cost, total revenue and profit into one column
procat = pd.melt(procat, id_vars="Product_Category", var_name="Metric", value_name="Total")
sns.barplot(data=procat, x="Product_Category", y="Total", hue="Metric");


# In[39]:


boxplot_function(dataframe = connecttel_df, x_value = "InternetService", y_value = "MonthlyCharges", palette = "YlGnBu")
plt.show()


# ## Observation:
# 
# 1. It could easily be seen from the box plots that the people who opted for 'Fiber optic' service have higher monthly charges.
# 2. People who opted for 'DSL' service has signifantly lower monthly charges as shown above.
# 3. As expecte, customers who do not enroll in the internet service have low charges as shown. 

# In[40]:


boxplot_function(dataframe = connecttel_df, x_value = "InternetService", y_value = "TotalCharges", palette = "YlGnBu")
plt.show()


# ## Observation:
# 
# 1. It could easily be seen from the box plots that the people who opted for 'Fiber optic' service have higher total charges.
# 2. People who opted for 'DSL' service has signifantly lower total charges as shown above.
# 3. As expecte, customers who do not enroll in the internet service have low charges as shown. 

# In[41]:


boxplot_function(dataframe = connecttel_df, x_value = "SeniorCitizen", y_value = "MonthlyCharges", palette = "YlGnBu")
plt.show()


# ## Observation:
# 
# 1. Monthly charges are significantly higher for Senior Citizens compared to Non-Senior Citizens respectively.
# 2. As a result, this leads us to believe that senior citizens are more inclined to add more services from Telco.
# 3. Therefore, Telco could take action and provide more interesting services to senior citizens compared to non-senior citizens.

# In[42]:


boxplot_function(dataframe = connecttel_df, x_value = "SeniorCitizen", y_value = "TotalCharges", palette = "YlGnBu")
plt.show()


# ## Observation:
# 
# 1. Based on the above boxplot, it could be seen that whether a person is a senior citizen or not has an impact of the total charges.
# 2. Senior citizens usually are quite rich and they usually work which means that they have higher income.
# 3. As a result, they might be opting for more services from Telco leading to higher total charges.

# In[44]:


barplot_function(dataframe = connecttel_df, x_value = "DeviceProtection", y_value = "TotalCharges", palette = "YlGnBu")
plt.show()


# ## Observation:
# 
# 1. Device Protection Plans have a very high cost as could be seen from the above.
# 2. This means that people are paying a lot for Device Protection plans.
# 3. We have seen from the previous plots that higher the charges, the more inclined are the customers to leave the Telco service.
# 4. Hence, Telco could take steps to reduce the prices for the Device Protection plans. 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # Multivariate Analysis
# Bivariate Analysis involves analyzing the relationship between two variables
# - Let us focus on Churn, which means we will be comparing varius colums, while using Churn as the hue
# 
# ## Categorical Columns
# - Churn vs gender
# - Churn vs Senior Citizen
# - Churn vs Partner
# - Churn vs Dependents
# - Churn vs Phone Service
# - Churn vs Multiple Lines
# - Churn vs Internet Service
# - Churn vs Online Security
# - Churn vs Online Backup
# - Churn vs Device Protection
# - Churn vs Tech Support
# - Churn vs Streaming TV
# - Churn vs Streaming Movies
# - Churn vs Contract
# - Churn vs Paperless Billing
# - Churn vs Total Charges
# ## Numerical Columns
# - Churn vs Payment Method
# - Churn vs Monthly Charges

# - Merchant
# - Port co

# In[71]:


connecttel_df.head()
churn_df


# ### Function 1
# def countplot_function(dataframe, column, figsize = (15, 10), palette = "BrBG"):
#     plt.figure(figsize = figsize)
#     sns.countplot(dataframe[column], palette = palette)
#     plt.title("{} countplot".format(column), fontsize = 15)
#     plt.xlabel("{}".format(column), fontsize = 10)
#     
# ### Use
# countplot_function(dataframe = connecttel_df, column = 'gender', palette = "autumn")
# plt.show()
#     

# ### Function 2
# def barplot_function(dataframe, x_value, y_value, title_size = 15, label_size = 10, figsize = (15,10), palette = "virdis"):
#     plt.figure(figsize = figsize)
#     sns.barplot(x = x_value, y = y_value, data = dataframe, palette = palette)
#     plt.xlabel("{} Value".format(x_value), fontsize = label_size)
#     plt.ylabel("{} Value".format(y_value), fontsize = label_size)
#     plt.title("{} Vs. {} Barplot".format(x_value, y_value), fontsize = title_size)
#     
#     
#     
# ### Use    
# barplot_function(dataframe = connecttel_df, x_value = "gender", y_value = "Churn", palette = "YlGnBu")
# #plt.savefig("Plots/InternetService Vs. MonthlyCharges.png")
# plt.show()

# ### Function3
# def barplot_function(dataframe, x_value, y_value, hue_value, title_size=15, label_size=10, figsize=(15, 10), palette="viridis"):
#     plt.figure(figsize=figsize)
#     sns.barplot(x=x_value, y=y_value, hue=hue_value, data=dataframe, palette=palette)
#     plt.xlabel("{} Value".format(x_value), fontsize=label_size)
#     plt.ylabel("{} Value".format(y_value), fontsize=label_size)
#     title = "{} Vs. {} Barplot".format(x_value, y_value)
#     if hue_value:
#         title += " (Hue: {})".format(hue_value)
#     plt.title(title, fontsize=title_size)
#     plt.show()
# 
# ### Use
# barplot_function(dataframe = connecttel_df, x_value = "InternetService", y_value = "MonthlyCharges", hue_value = "Churn", palette = "YlGnBu")
# plt.show()

# In[46]:


def barplot_function(dataframe, x_value, y_value, hue_value, title_size=15, label_size=10, figsize=(15, 10), palette="viridis"):
    plt.figure(figsize=figsize)
    sns.barplot(x=x_value, y=y_value, hue=hue_value, data=dataframe, palette=palette)
    plt.xlabel("{} Value".format(x_value), fontsize=label_size)
    plt.ylabel("{} Value".format(y_value), fontsize=label_size)
    title = "{} Vs. {} Barplot".format(x_value, y_value)
    if hue_value:
        title += " (Hue: {})".format(hue_value)
    plt.title(title, fontsize=title_size)
    plt.show()

# Example usage:
# barplot_function(your_dataframe, 'x_column', 'y_column', hue_value='hue_column')


# In[161]:


barplot_function(dataframe = connecttel_df, x_value = "InternetService", y_value = "MonthlyCharges", hue_value = "Churn", palette = "YlGnBu")
#plt.savefig("Plots/InternetService Vs. MonthlyCharges.png")
plt.show()


# # Observation
# - DSL Customers Churn Less: The "No churn" bar for DSL is the tallest, suggesting customers with DSL internet service are less likely to churn compared to those with fiber optic or no internet service.
# - Fiber Optic Has Most Churn: Conversely, the "Churn" bar for fiber optic is the tallest, indicating a higher churn rate among customers with this service compared to DSL and no internet.
# - No Internet Has Mixed Results: The churn rate for customers with no internet service falls between DSL and fiber optic, with a relatively even split between "Churn" and "No churn" bars.
# -  Charge Variations: Within each internet service category, churn rates appear to vary based on monthly charges. Higher charges generally correspond to lower churn rates, though the trend isn't always consistent.

# In[47]:


barplot_function(dataframe = connecttel_df, x_value = "InternetService", y_value = "TotalCharges", hue_value = "Churn", palette = "YlGnBu")
#plt.savefig("Plots/InternetService Vs. MonthlyCharges.png")
plt.show()


# ## Observation:
# 
# 1. Based on this plot, we see that a large portion of customers from Fiber optic option tend to leave the service compared to other internet services.
# 2. Other services such as DSL service have higher number of customers who are willing to stay with the service.
# 3. Therefore, Telco might consider what might be the potential case for customers who have taken fiber option to leave the service.
# 4. If they could come up with the right tactics to improve their fiber option service, this ensures that a large portion of customers are retained. 

# In[145]:


connecttel_df['gender'] = connecttel_df['gender'].replace({'Male': 1, 'Female': 2})
connecttel_df['Churn'] = connecttel_df['Churn'].replace({'Yes': 1, 'No': 0})

#churn_df['Dependents'] = churn_df['Dependents'].replace({'Yes': 1, 'No': 0})
#churn_df['Churn'] = churn_df['Churn'].replace({'Yes': 1})
#churn_df


# In[48]:


barplot_function(dataframe = connecttel_df, x_value = "Contract", y_value = "MonthlyCharges", hue_value = "Churn", palette = "YlGnBu")
plt.show()


# # Observation
# - Contract Type as a Strong Retention Factor: The observed differences in churn rates across contract types highlight the importance of contract length as a retention strategy. Encouraging longer-term commitments could potentially reduce churn.
# - Complex Relationship with Charges: The interplay between monthly charges and churn within each contract type suggests that pricing strategies might need to be tailored based on contract length to effectively manage churn.

# In[49]:


barplot_function(dataframe = connecttel_df, x_value = "Contract", y_value = "TotalCharges", hue_value = "Churn", palette = "YlGnBu")
plt.show()


# ## Observation
# 
# - Month-to-Month Retains More: The "No churn" bar for month-to-month contracts is taller than the "Yes churn" bar, indicating a higher proportion of customers sticking with month-to-month plans compared to those who churn. 
# - Longer Contracts Churn Less: The "Yes churn" bars remain generally taller than "No churn" bars for one-year and two-year contracts.
# - Month-to-month appears to have a relatively higher customer retention rate compared to longer-term contracts. Therefore, ConnectTel should improve on the month-to-month contract package's total charge

# In[63]:


barplot_function(dataframe = connecttel_df, x_value = "StreamingTV", y_value = "MonthlyCharges", hue_value = "Churn", palette = "YlGnBu")
plt.show()


# In[ ]:





# In[50]:


barplot_function(dataframe = connecttel_df, x_value = "gender", y_value = "TotalCharges", hue_value = "Churn", palette = "YlGnBu")
#plt.savefig("Plots/InternetService Vs. MonthlyCharges.png")
plt.show()


# ## Observation
# - Gender does not reveal any influence on customer churn, seeing that the proportion of churn and no churn are equal with both male and female. 

# In[115]:


### Function3
def barplot_function(dataframe, x_value, y_value, hue_value, title_size=15, label_size=10, figsize=(15, 10), palette="viridis"):
    plt.figure(figsize=figsize)
    sns.barplot(x=x_value, y=y_value, hue=hue_value, data=dataframe, palette=palette)
    plt.xlabel("{} Value".format(x_value), fontsize=label_size)
    plt.ylabel("{} Value".format(y_value), fontsize=label_size)
    title = "{} Vs. {} Barplot".format(x_value, y_value)
    if hue_value:
        title += " (Hue: {})".format(hue_value)
    plt.title(title, fontsize=title_size)
    plt.show()


# In[181]:


barplot_function(dataframe = connecttel_df, x_value = "PhoneService", y_value = "tenure", hue_value = "Churn", palette = "YlGnBu")
plt.show()


# ## Observation
# - Higher Customer Retention in All Groups: The data suggests that the telecommunications company enjoys a relatively high overall customer retention rate across both phone service categories and different tenure lengths. This is a positive finding and indicates effectiveness in keeping customers engaged and satisfied with their service.

# In[69]:


barplot_function(dataframe = connecttel_df, x_value = "PhoneService", y_value = "tenure", hue_value = "MultipleLines", palette = "spring")
plt.show()


# ## Observe
# - Customers who have phone service with multiple lines have a longer tenure that those without multiple lines.

# In[71]:


barplot_function(dataframe = connecttel_df, x_value = "PhoneService", y_value = "MonthlyCharges", hue_value = "Churn", palette = "YlGnBu")
plt.show()


# ## Observation
# - Customers with phone service tend to churn more based on the monthly charges, while customers without phone service churn less because their monthly charges are less.

# In[65]:


barplot_function(dataframe = connecttel_df, x_value = "PhoneService", y_value = "TotalCharges", hue_value = "Churn", palette = "YlGnBu")
plt.show()


# ## Observation
# - Customers with phone service have a sugnificantly lower "No Churn" that "Yes Churn" based on the total charges. The same applies to customers without phone service.

# In[72]:


barplot_function(dataframe = connecttel_df, x_value = "OnlineSecurity", y_value = "TotalCharges", hue_value = "Churn", palette = "YlGnBu")
plt.show()


# In[ ]:





# In[73]:


barplot_function(dataframe = connecttel_df, x_value = "OnlineSecurity", y_value = "MonthlyCharges", hue_value = "Churn", palette = "YlGnBu")
plt.show()


# In[ ]:





# In[74]:


barplot_function(dataframe = connecttel_df, x_value = "OnlineSecurity", y_value = "tenure", hue_value = "Churn", palette = "YlGnBu")
plt.show()


# In[ ]:





# In[75]:


barplot_function(dataframe = connecttel_df, x_value = "OnlineBackup", y_value = "MonthlyCharges", hue_value = "Churn", palette = "YlGnBu")
plt.show()


# In[76]:


barplot_function(dataframe = connecttel_df, x_value = "OnlineBackup", y_value = "TotalCharges", hue_value = "Churn", palette = "YlGnBu")
plt.show()


# In[ ]:





# honeService: Understanding if having phone service impacts churn.
# MultipleLines: Examining whether multiple lines are associated with higher or lower churn rates.
# InternetService: Analyzing the effect of different internet service types (DSL, Fiber optic, or none) on churn.
# OnlineSecurity: Exploring the relationship between online security subscription and churn.
# OnlineBackup: Determining if online backup service usage affects churn.
# DeviceProtection: Assessing the impact of device protection on churn.
# TechSupport: Examining the association between tech support usage and churn.
# StreamingTV: Investigating the link between streaming TV subscription and churn.
# StreamingMovies: Assessing the impact of streaming movies on churn.

# In[122]:


barplot_function(dataframe = connecttel_df, x_value = "StreamingMovies", y_value = "tenure", hue_value = "Churn", palette = "YlGnBu")
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## Heatmap
# 
# Heatmaps are a good way to visualize our data and understanding trends from it. In our case, we consider the correlation between all the set of features and compute the correlation coefficients. But interpreting the numerical vector (correlation coefficient vector) can be tedious and quite difficult to comprehend. Therefore, we will now be using Heatmap plots which generate brightened colors if they find a particular value to be high or vice-versa. The reverse might also be true depending on the 'cmap' or 'palette' that we select for our heatmaps. 

# In[13]:


df_categorical = connecttel_df.select_dtypes(include = "object")


# In[14]:


df_numerical = connecttel_df.select_dtypes(exclude = "object")


# #### Displaying the columns in our data that are categorical in nature. 

# In[15]:


print("The columns that are numerical in our data are:\n {}".format(df_numerical.columns))


# ##### Displaying the head or the first 5 rows in our dataframe. 

# In[16]:


df_categorical.head()


# In[17]:


df_numerical.head()


# In[18]:


df_categorical.head()


# ##### Removing the feature that do not add a lot of meaning for our machine learning predictions. 

# In[19]:


df_categorical.drop(['customerID'], axis = 1, inplace = True)


# ##### Performing the one hot encoding of the categorical features present in our dataset. 

# In[20]:


pd.get_dummies(df_categorical.gender, drop_first = True).head()


# In[21]:


df_dummy_encoding = pd.get_dummies(df_categorical, drop_first = True)


# In[22]:


connecttel_df_final = pd.concat([df_dummy_encoding, df_numerical], axis = 1)


# In[23]:


connecttel_df_final.head()


# In[24]:


connecttel_df.corr()


# In[24]:


plt.figure(figsize = (15, 15))
sns.heatmap(connecttel_df_final.corr(), annot = True)
plt.show()


# ## Visuals Based on The Key Correlations 

# In[126]:


barplot_function(dataframe = connecttel_df, x_value = "StreamingMovies", y_value = "MonthlyCharges", hue_value = "Churn", palette = "YlGnBu")
plt.show()


# In[127]:


barplot_function(dataframe = connecttel_df, x_value = "StreamingTV", y_value = "MonthlyCharges", hue_value = "Churn", palette = "YlGnBu")
plt.show()


# In[136]:


# Sort the DataFrame by 'Tenure' for a chronological order
connecttel_df.sort_values(by='tenure', inplace=True)

# Create a multivariate line chart with 'Churn' as the hue
plt.figure(figsize=(10, 6))
sns.lineplot(x='tenure', y='TotalCharges', hue='Churn', data=connecttel_df, marker='o')

# Customize the plot
plt.title('Trend of Customer Churn with Churn as Hue')
plt.xlabel('Tenure')
plt.ylabel('Your Numerical Column')
plt.legend(title='Churn', loc='best')
plt.grid(True)
plt.show()


# In[137]:


connecttel_df['InternetService'].unique()


# In[138]:


# Divide the dataset into two based on 'Internet Service'
connecttel_df_FibreOptic = connecttel_df[connecttel_df['InternetService'] == 'Fiber optic']
connecttel_df_DSL = connecttel_df[connecttel_df['InternetService'] == 'DSL']

# Calculate the sum from the numerical column for each subset
sum_FO = connecttel_df_FibreOptic['MonthlyCharges'].sum()
sum_DSL = connecttel_df_DSL['MonthlyCharges'].sum()

print(f"Sum for Monthly Charges 'Fiber Optic': {sum_FO}")
print(f"Sum for Monthly Charges 'DSL': {sum_DSL}")


# In[139]:


barplot_function(dataframe = connecttel_df_FibreOptic, x_value = "InternetService", y_value = "MonthlyCharges", hue_value = "Churn", palette = "YlGnBu")
plt.show()


# ## Observation:
# 
# #### Streaming Movies 'Yes' and Streaming TV 'Yes' have a moderately positive relationship with Monthly Charges (0.63):  
# - The value of 0.63 being positive signifies that as the tendency of a customer to stream movies and stream TV increases, their monthly charge also tends to increase.
# 
# - However, the value of 0.63 isn't close to 1, which would represent a perfect correlation. This means that while there's a noticeable trend, it's not absolute. There might be customers who stream movies and TV frequently but have lower charges, and vice versa. Perhaps customers who stream movies and TV more often have higher subscription tiers, pay for additional movie rentals, or simply generate higher data usage due to streaming, leading to increased charges.
# ##### Solutions: 
# - Consider targeted marketing campaigns towards those customer segments to encourage further engagement with streaming services. 
# - Explore options like data compression technologies or introducing capped plans to manage costs while still offering a good streaming experience. 
# - Consider personalized recommendations, targeted promotions, and tailored data plans, potentially improving customer satisfaction and reducing churn.
# 
# 
# #### Tenure have a high positive relationship with Total Charges (0.83): 
# - Based on the heatmap given, it could be seen that there is a strong correlation between the TotalCharges and the Tenure of staying in a service. Customers who stay for a long time might become more accustomed to and reliant on your services, making them less price-sensitive and open to spending more. Also, Over time, customers might use your services more frequently or expand their plan options, leading to higher charges.
# ##### Solutions: 
# - Leverage this insight to develop targeted retention campaigns for different tenure segments. Offer attractive incentives and value propositions to maintain high-spending, long-term customers while encouraging spending increases in lower-tenured segments. 
# - Focus on identifying and addressing potential churn triggers for shorter-tenured customers. Understanding why some customers leave early can help you implement improvements or targeted offers to increase customer satisfaction and encourage longer stays.
# - Consider implementing pricing models that reward customer loyalty and encourage increased spending based on tenure. This could involve tiered discounts, loyalty points, or exclusive benefits for longer-standing customers.
# - Ensure a consistently positive customer experience throughout the entire lifecycle, not just during initial acquisition. By building strong relationships and delivering valuable services, you can keep customers satisfied and incentivize them to stay longer and spend more.
# 
# #### Internet Service (Fibre OPtic) users have a high positive relationship with Monthly Charges (0.79): 
# - As the value is positive, it signifies that customers who opt for Fibre Optic internet tend to have higher monthly charges compared to those with other service types. Therefore, choosing Fibre Optic is significantly associated with a likely increase in monthly charges.
# ##### Solutions:
# - Clearly communicate the benefits and costs of Fibre Optic compared to other options. Highlight the value proposition of the higher speeds and bundled services to justify the price point and avoid customer dissatisfaction.
# - Focus marketing efforts on segments who would appreciate the advantages of Fibre Optic, like gamers, streamers, or businesses with high data needs. This ensures you reach customers willing to pay for the premium service.
# - Focus marketing efforts on segments who would appreciate the advantages of Fibre Optic, like gamers, streamers, or businesses with high data needs. This ensures you reach customers willing to pay for the premium service.
# - Consider offering tiered pricing plans for Fibre Optic with different bandwidth levels and service combinations. This caters to diverse user needs and budgets, attracting a wider customer base while still capturing the value potential of Fibre Optic.
# - Provide customers with tools or options to manage their data usage, such as usage alerts, data caps, or flexible plan upgrades. This empowers customers to optimize their spending and avoid bill surprises.
# 
# 

# # DATA PREPROCESSING

# In[25]:


X = connecttel_df_final.drop(['Churn_Yes'], axis = 1)
Y = connecttel_df_final['Churn_Yes']


# ## Train Test Split
# 
# It is now time to split our data into training and cross-validation data. This is because, we need to test whether our model is performing well on the unseen class (cross-validation and test data). We need to use cross-validation data to tune the hyperparameters and improve the performance of the best machine learning models.
# 
# We are going to be dividing the data in such a way whether the training data contains around 70% of the records and the test data contains around 30% of the records respectively. Below the code that implements these steps. 

# In[26]:


X_train, X_cv, Y_train, Y_cv  = train_test_split(X, Y, test_size = 0.3, random_state = 101)


# ## Standardization
# 
# When we take data that is present in many scales, it can sometimes make the machine leanring models to assume that a particular feature is more important due to the fact that it is of an entirely different scale and so on. Therefore, we need to ensure that the entire data is of the same scale so that all the features are treated equally without giving more importance to one particular feature than the other. 
# 
# We perform standardization by taking the mean of a particular feature and subtracting from all the values from that particular feature and divide the results from the standard deviation of the same feature itself. This method is continued for all the features until each and every feature has a mean of 0 and a standard deviation of 1. Below are the set of steps that are performed to standardize the features. 

# In[27]:


from sklearn.preprocessing import StandardScaler


# In[28]:


scaler = StandardScaler()
scaler.fit(X_train)
X_train_transformed = scaler.transform(X_train)
X_cv_transformed = scaler.transform(X_cv)


# In[29]:


len(X_train_transformed)


# In[30]:


len(X_cv_transformed)


# In[31]:


from sklearn.decomposition import PCA


# In[32]:


pca = PCA(n_components = 30)
pca.fit(X_train_transformed)
X_train_reduced = pca.transform(X_train_transformed)
X_cv_reduced = pca.transform(X_cv_transformed)


# In[33]:


X_train_reduced.shape


# In[34]:


X_train_transformed.shape


# In[35]:


principal_components = list(range(1, len(pca.explained_variance_ratio_) + 1))


# In[36]:


plt.figure(figsize=(20, 10))
sns.lineplot(x=principal_components, y=np.cumsum(pca.explained_variance_ratio_), color='orange')
plt.xlabel('Principal Components', fontsize=15)
plt.ylabel('Variance Explained', fontsize=15)
plt.title("The Percentage of Variance Explained", fontsize=20)
plt.show()


# ##### Printing the shape of the dataframe to get an understanding of the number of rows and columns we would be dealing with for our data. 

# In[37]:


print(len(X_train_transformed))


# ##### Printing the shape of the cross-validation dataframe to get an understanding of the number of rows and columns we would be dealing with for our data. 

# In[38]:


print(len(X_cv_transformed))


# ##### Checking to see for the training data whether we have equal number of classes. But it could be seen that there is class imbalance as we have seen earlier in the plots as well. In the next few cells, we would be performing class balancing so that the ML models learn to represent both the classes of interest with equal probability. 

# In[39]:


Y_train.value_counts()


# ##### Similarly, we can also take a look at the cross-validation data and find out the total number of classes and the count of them to see if there is a class imbalance respectively.

# In[45]:


Y_cv.value_counts()


# In[40]:


len(X_train_transformed[0])


# In[47]:


#!pip install -U imbalanced-learn


# ## Class Balancing
# 
# One of the most popular methods to perform class balancing is with SMOTE. We will initialize it and fit our training and cross-validation data points which ensure that we have equal number of classes which we are interested to work and proceed. Below is the code sample that exactly shows all these steps.
# 

# In[41]:


from imblearn.over_sampling import SMOTE


# In[42]:


sampler = SMOTE()
X_train_transformed, Y_train = sampler.fit_resample(X_train_transformed, Y_train)
X_cv_transformed, Y_cv = sampler.fit_resample(X_cv_transformed, Y_cv)


# ##### After performing the class balancing, we are now going to fit our data with the K Neighbors Classifier and understand the performance of the models and their predictions respectively.

# In[43]:


model = KNeighborsClassifier()
model.fit(X_train_transformed, Y_train)
Y_predictions = model.predict(X_cv_transformed)


# ## Confusion Matrix
# 
# A confusion matrix would give us the values between the actual and the predicted outcomes. If we have a 2 class problem, we would be getting a 2 * 2 matrix that would basically give us the predictions and the actual values respectively. 

# In[44]:


print(confusion_matrix(Y_cv, Y_predictions))


# ## Classification Report
# 
# Classification report gives a very good understanding of the precision, recall and the f1-score along with accuracy that are important metrics that we generally consider when measuring the performance of a general classifier model. Therefore, we would be taking into account the classification report and checking to see the values and comparing them with other models in order to pick the best model for our task for predicting whether a customer is going to churn or not.
# 

# In[45]:


from sklearn.metrics import classification_report


# In[46]:


print(classification_report(Y_cv, Y_predictions))


# ## Plot Confusion Matrix
# 
# Rather than taking only a look at the numerical values, it is better to plot the confusion matrix along with their color coded outputs so that we can better access the performance of the classifier model at hand. 

# In[47]:


pip install -U scikit-learn


# In[57]:


print(sklearn.__version__)


# In[49]:


from sklearn.metrics import confusion_matrix


# In[ ]:


# from sklearn.metrics import plot_confusion_matrix


# In[55]:


import matplotlib.pyplot as plt
import sklearn


# In[58]:


Y_cv_pred = model.predict(X_cv_transformed)

cm = confusion_matrix(Y_cv, Y_cv_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="PuBuGn", linewidths=.5)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# ## ROC AUC Curves
# 
# We can also understand the performance of the model by taking into account the AUC (Area under the curve) and checking if the values is close to 1. The higher is the value of AUC, the better the model is said to be performing and vice-versa. Below is the plot to get an understanding of the performance of the ML model. 

# In[60]:


## https://medium.com/analytics-vidhya/evaluation-metrics-for-classification-problems-with-implementation-in-python-a20193b4f2c3
## Credits to this website for providing such an useful and clear explanation of the topic along with the code snippet 
## which was modified for our application

from sklearn.metrics import roc_curve, auc

class_probabilities = model.predict_proba(X_cv_transformed)
preds = class_probabilities[:, 1]

fpr, tpr, threshold = roc_curve(Y_cv, preds)
roc_auc = auc(fpr, tpr)

# Printing AUC
print(f"AUC for our classifier is: {roc_auc}")

# Plotting the ROC
plt.figure(figsize = (10, 10))
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
#plt.savefig('Machine Learning Plots/KNN AUC Plot.png')
plt.show()


# ## Support Vector Classifier
# 
# Support vector classifier is also a popular machine learning classifier that can be used to predict whether a customer is going to churn from a service or not. A support vector machine (SVM) is machine learning algorithm that analyzes data for classification and regression analysis. SVM is a supervised learning method that looks at data and sorts it into one of two categories. An SVM outputs a map of the sorted data with the margins between the two as far apart as possible. SVMs are used in text categorization, image classification, handwriting recognition and in the sciences.
# 
# Source: https://www.techopedia.com/definition/30364/support-vector-machine-svm
# 

# In[62]:


model = SVC(probability = True)
model.fit(X_train_transformed, Y_train)
Y_predictions = model.predict(X_cv_transformed)


# ## Confusion Matrix
# 
# Let us now check the confusion matrix to find out how our model is performing on the cross-validation data. 

# In[63]:


print(confusion_matrix(Y_cv, Y_predictions))


# ## Classification Report
# 
# We can now take a look at a few metrics and understand the performance of the Support Vector Classifier. We consider values from metrics such as precision, recall, f1-score along with accuracy to access the model performance. 

# In[64]:


print(classification_report(Y_cv, Y_predictions))


# ## Plot Confusion Matrix
# 
# Let us know plot the confusion matrix to see the performance of the SVM classifier respectively. Below is the plot that shows the output based on the actual values and the predicted values. 

# In[ ]:


# will the result of this code below
Y_cv_pred = model.predict(X_cv_transformed)
cm = confusion_matrix(Y_cv, Y_cv_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="PuBuGn", linewidths=.5)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# And this code below
fig, ax = plt.subplots(figsize = (10, 10))
plot_confusion_matrix(model, X_cv_transformed, Y_cv, ax = ax, cmap = 'PuBuGn')
plt.grid(False)
plt.show()


# In[66]:


Y_cv_pred = model.predict(X_cv_transformed)

cm = confusion_matrix(Y_cv, Y_predictions)

plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="magma", linewidths=.5)
plt.title("Confusion Matrix plot", fontsize=15)
plt.grid(False)
plt.show()


# ## ROC AUC Curves
# 
# We can now consider the AUC and check the performance of the model. One thing to note is that the higher the value of AUC, the better is the model performing on the cross-validation data. Let us now see the performance of the support vector classifier. 

# In[68]:


## https://medium.com/analytics-vidhya/evaluation-metrics-for-classification-problems-with-implementation-in-python-a20193b4f2c3
## Credits to this website for providing such an useful and clear explanation of the topic along with the code snippet
## which was modified for our application. 

from sklearn.metrics import roc_curve, auc

class_probabilities = model.predict_proba(X_cv_transformed)
preds = class_probabilities[:, 1]

fpr, tpr, threshold = roc_curve(Y_cv, preds)
roc_auc = auc(fpr, tpr)

# Printing AUC
print(f"AUC for our classifier is: {roc_auc}")

# Plotting the ROC
plt.figure(figsize = (8, 8))
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
#plt.savefig('Machine Learning Plots/SVM AUC Curves.png')
plt.show()


# ## Logistic Regression

# In[69]:


from sklearn.linear_model import LogisticRegression


# In[71]:


model = LogisticRegression()
model.fit(X_train_transformed, Y_train)
Y_predictions = model.predict(X_cv_transformed)


# In[72]:


print(confusion_matrix(Y_cv, Y_predictions))


# In[73]:


print(classification_report(Y_cv, Y_predictions))


# In[75]:


Y_cv_pred = model.predict(X_cv_transformed)
cm = confusion_matrix(Y_cv, Y_cv_pred)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="cividis", linewidths=.5)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# In[77]:


## https://medium.com/analytics-vidhya/evaluation-metrics-for-classification-problems-with-implementation-in-python-a20193b4f2c3
## Credits to this website for providing such an useful and clear explanation of the topic along with the code snippet
## which was modified for our application. 

from sklearn.metrics import roc_curve, auc

class_probabilities = model.predict_proba(X_cv_transformed)
preds = class_probabilities[:, 1]

fpr, tpr, threshold = roc_curve(Y_cv, preds)
roc_auc = auc(fpr, tpr)

# Printing AUC
print(f"AUC for our classifier is: {roc_auc}")

# Plotting the ROC
plt.figure(figsize = (7,7))
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
#plt.savefig('Machine Learning Plots/Logistic Regression AUC.png')
plt.show()


# ## Decision Tree Classifier

# In[78]:


model = DecisionTreeClassifier()
model.fit(X_train_transformed, Y_train)
Y_predictions = model.predict(X_cv_transformed)


# In[79]:


print(confusion_matrix(Y_cv, Y_predictions))


# In[80]:


print(classification_report(Y_cv, Y_predictions))


# In[81]:


Y_cv_pred = model.predict(X_cv_transformed)
cm = confusion_matrix(Y_cv, Y_cv_pred)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="inferno", linewidths=.5)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# In[82]:


## https://medium.com/analytics-vidhya/evaluation-metrics-for-classification-problems-with-implementation-in-python-a20193b4f2c3
## Credits to this website for providing such an useful and clear explanation of the topic along with the code snippet
## which was modified for our application. 

from sklearn.metrics import roc_curve, auc

class_probabilities = model.predict_proba(X_cv_transformed)
preds = class_probabilities[:, 1]

fpr, tpr, threshold = roc_curve(Y_cv, preds)
roc_auc = auc(fpr, tpr)

# Printing AUC
print(f"AUC for our classifier is: {roc_auc}")

# Plotting the ROC
plt.figure(figsize = (7, 7))
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
#plt.savefig("Machine Learning Plots/Decision Tree AUC actual.png")
plt.show()


# ## Gaussian Naive Bayes

# In[83]:


model = GaussianNB()
model.fit(X_train_transformed, Y_train)
Y_predictions = model.predict(X_cv_transformed)


# In[85]:


print(confusion_matrix(Y_cv, Y_predictions))


# In[86]:


print(classification_report(Y_cv, Y_predictions))


# In[87]:


Y_cv_pred = model.predict(X_cv_transformed)

cm = confusion_matrix(Y_cv, Y_cv_pred)

plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="OrRd", linewidths=.5)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# In[88]:


## https://medium.com/analytics-vidhya/evaluation-metrics-for-classification-problems-with-implementation-in-python-a20193b4f2c3
## Credits to this website for providing such an useful and clear explanation of the topic along with the code snippet
## which was modified for our application. 

from sklearn.metrics import roc_curve, auc

class_probabilities = model.predict_proba(X_cv_transformed)
preds = class_probabilities[:, 1]

fpr, tpr, threshold = roc_curve(Y_cv, preds)
roc_auc = auc(fpr, tpr)

# Printing AUC
print(f"AUC for our classifier is: {roc_auc}")

# Plotting the ROC
plt.figure(figsize = (7, 7))
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
#plt.savefig("Machine Learning Plots/Naive Bayes AUC Curves.png")
plt.show()


# ## Random Forest Classifier

# In[89]:


model = RandomForestClassifier()
model.fit(X_train_transformed, Y_train)
Y_predictions = model.predict(X_cv_transformed)


# In[90]:


print(confusion_matrix(Y_cv, Y_predictions))


# In[ ]:


print(classification_report(Y_cv, Y_predictions))


# In[91]:


Y_cv_pred = model.predict(X_cv_transformed)
cm = confusion_matrix(Y_cv, Y_cv_pred)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="magma", linewidths=.5)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# In[92]:


## https://medium.com/analytics-vidhya/evaluation-metrics-for-classification-problems-with-implementation-in-python-a20193b4f2c3
## Credits to this website for providing such an useful and clear explanation of the topic along with the code snippet
## which was modified for our application. 

from sklearn.metrics import roc_curve, auc

class_probabilities = model.predict_proba(X_cv_transformed)
preds = class_probabilities[:, 1]

fpr, tpr, threshold = roc_curve(Y_cv, preds)
roc_auc = auc(fpr, tpr)

# Printing AUC
print(f"AUC for our classifier is: {roc_auc}")

# Plotting the ROC
plt.figure(figsize = (7, 7))
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
#plt.savefig("Machine Learning Plots/Random Forests AUC.png")
plt.show()


# ## Xgb Classifier

# In[94]:


get_ipython().system('pip install xgboost')


# In[96]:


import xgboost as xgb


# In[97]:


model = xgb.XGBClassifier()
model.fit(X_train_transformed, Y_train)
Y_predictions = model.predict(X_cv_transformed)


# In[98]:


print(confusion_matrix(Y_cv, Y_predictions))


# In[99]:


print(classification_report(Y_cv, Y_predictions))


# In[100]:


Y_cv_pred = model.predict(X_cv_transformed)
cm = confusion_matrix(Y_cv, Y_cv_pred)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="OrRd", linewidths=.5)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# In[102]:


## https://medium.com/analytics-vidhya/evaluation-metrics-for-classification-problems-with-implementation-in-python-a20193b4f2c3
## Credits to this website for providing such an useful and clear explanation of the topic along with the code snippet
## which was modified for our application. 

from sklearn.metrics import roc_curve, auc

class_probabilities = model.predict_proba(X_cv_transformed)
preds = class_probabilities[:, 1]

fpr, tpr, threshold = roc_curve(Y_cv, preds)
roc_auc = auc(fpr, tpr)

# Printing AUC
print(f"AUC for our classifier is: {roc_auc}")

# Plotting the ROC
plt.figure(figsize = (7, 7))
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
#plt.savefig("Machine Learning Plots/XGB AUC Curve.png")
plt.show()


# ## Observation:
# 1. It could be seen from the above plots that the XBG model is performing better than all the remaining models that we have tried and tested as shown above. 
# 2. Therefore, we will now use our XGB model and perform hyperparameter tuning to get the best results and improving the AUC even further. 

# ## Hyperparameter Tuning of XGB model

# In[103]:


sub_sample = [0.1, 0.2, 0.3, 0.5, 0.7, 1]


# In[105]:


for sample_value in sub_sample:
    model = xgb.XGBClassifier(subsample = sample_value, eval_metric = 'logloss', n_jobs = -1)
    model.fit(X_train_transformed, Y_train)        
    Y_predictions = model.predict(X_cv_transformed)
    print("The percentage of samples used in XGB model = {}%".format(np.round(sample_value * 100, 2)))
    print(classification_report(Y_predictions, Y_cv))
    print("\n")


# In[106]:


max_depth_values = [1, 2, 3, 5, 10, 50, 100]

for depth in max_depth_values:
    model = xgb.XGBClassifier(max_depth = depth, eval_metric = 'logloss', n_jobs = -1)
    model.fit(X_train_transformed, Y_train)
    print("The depth of the trees = {}".format(depth, 2))
    Y_predictions = model.predict(X_cv_transformed)
    print(classification_report(Y_predictions, Y_cv))
    print("\n")


# In[107]:


num_estimator_values = [1, 2, 5, 10, 20, 50, 100, 500]
for estimators in num_estimator_values:
    model = xgb.XGBClassifier(n_estimators = estimators, eval_metric = 'logloss', n_jobs = -1)
    model.fit(X_train_transformed, Y_train)
    Y_predictions = model.predict(X_cv_transformed)
    print("The total number of estimators used in XGB model = {}".format(estimators))
    print(classification_report(Y_predictions, Y_cv))
    print("\n")


# In[108]:


best_number_of_estimators = 100
best_subsamples = 1
best_max_depth_values = 2

model = xgb.XGBClassifier(n_estimators = best_number_of_estimators, max_depth = best_max_depth_values, 
                         subsample = best_subsamples, eval_metric = 'logloss', n_jobs = -1)
model.fit(X_train_transformed, Y_train)
Y_predictions = model.predict(X_cv_transformed)
print(classification_report(Y_predictions, Y_cv))


# In[109]:


print(confusion_matrix(Y_cv, Y_predictions))


# In[110]:


Y_cv_pred = model.predict(X_cv_transformed)

cm = confusion_matrix(Y_cv, Y_cv_pred)

plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrBr", linewidths=.5)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# In[111]:


## https://medium.com/analytics-vidhya/evaluation-metrics-for-classification-problems-with-implementation-in-python-a20193b4f2c3
## Credits to this website for providing such an useful and clear explanation of the topic along with the code snippet
## which was modified for our application. 

from sklearn.metrics import roc_curve, auc

class_probabilities = model.predict_proba(X_cv_transformed)
preds = class_probabilities[:, 1]

fpr, tpr, threshold = roc_curve(Y_cv, preds)
roc_auc = auc(fpr, tpr)

# Printing AUC
print(f"AUC for our classifier is: {roc_auc}")

# Plotting the ROC
plt.figure(figsize = (7, 7))
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
#plt.savefig("Machine Learning Plots/Random Forests Hyperparameter Tuned ROC.png")
plt.show()


# ## Model Evaluation and Interpretation

# In[112]:


from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


# In[113]:


print("The precision score of the best XGB model is: {:.2f}".format(precision_score(Y_predictions, Y_cv)))
print("The recall score of the best XGB model is: {:.2f}".format(recall_score(Y_predictions, Y_cv)))
print("The f1 score of the best XGB model is: {:.2f}".format(f1_score(Y_predictions, Y_cv)))
print("The micro f1 score of the best XGB model is: {:.2f}".format(f1_score(Y_predictions, Y_cv, average = 'micro')))
print("The macro f1 score of the best XGB model is: {:.2f}".format(f1_score(Y_predictions, Y_cv, average = 'macro')))
print("The accuracy of the best XGB model is: {:.2f}%".format(accuracy_score(Y_predictions, Y_cv) * 100))


# ## Observation: 
# 
# 1. It could be seen based on the results that the model that we have chosen and __hyperparameter tuned (XGboost)__ is performing really well on the cross-validation data. 
# 2. Therefore, the model would be able to predict whether a customer is going to __churn (leave the service) or not__ with an accuracy of about __85%__ which is not bad for a model that contains imbalanced data along with less number of samples that is completely contrary to a real-world problem that contains millions of customers or data points respectively. 
# 3. Therefore, we can expect similar performance of the model on the __test data__ as well provided that the distribution of the data does not change compared to the __cross-validation data__. 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## K Neighbors Classifier
# 
# Since the output variable that we are going to be predicting is discrete, we are going to be using various machine learning classifiers for our task of predicting whether a customer is going to churn from a service or not.
# 
# Note: We would be first performing class balancing which can be done with the library "imbalanced-learn" that can also be dowloaded using simple "pip" or "conda" commands. Below is the cell you might uncomment and run if you do not have this library installed already.
# 

# In[ ]:





# In[ ]:





# ## 1. Problem definition: clearly articulate the problem that is to be solved with your data mining. How will the company benefit from your solution?
# - Discover the cause of customer churn and 
# - Implement customer retention initiatives to reduce attrition and enhance customer loyalty.
# 
# 
# 
# ## 2. Perform exploratory data analysis in Python:
# 
# - a) Visualize relationships between the label and some key features
# - b) Explore correlations
# - c) Conduct univariate, bivariate, and multivariate analysis as much as is feasible
# 
# ## 3. Perform feature engineering:
# - a) Encoding categorical variables
# - b) Create new features from existing features where necessary, depending on insights from your EDA
# 
# ## 4. Model selection, training, and validation:
# 
# - a) Train and test at least 3 supervised learning model
# 
# ## 5. Model evaluation:
# 
# - a) Analyze the results of your trained model
# - b) What metrics are most important for the problem? Should the business be more concerned with better results on false negatives or true positives?
# 
# ## 6. Submission:
# 
# - a) Publish your Jupyter Notebook to your GitHub profile and Google Classroom.
# - b) In the readme file, include a description of the project and summarize the steps you took and the challenges you faced.
# - c) share the link with your instructor.

# In[ ]:




