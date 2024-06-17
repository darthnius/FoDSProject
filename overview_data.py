import math
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import pandas as pd
import scipy.stats as sts
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import t, ttest_ind
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc


data_feats = pd.read_csv("../data/driams_Escherichia coli_Ceftriaxone_features.csv")
data_labels = pd.read_csv("../data/driams_Escherichia coli_Ceftriaxone_labels.csv")
print(data_feats.head(5))
print(data_labels.head(5))
print(data_feats.shape)
print(data_labels.shape)
print(data_labels['Unnamed: 0'].nunique()) #no multiple entries of same sample ID
print(data_labels['label'].value_counts())

#How many patients do have a resistance
plt.figure(figsize=(10, 5))
count = sns.countplot(data_labels, x='label')
labels = ['No Resistance', 'Resistance']
count.set_xticklabels(labels)
plt.title('Distribution of Antibiotic Resistance')
plt.xlabel('Resistance')
plt.ylabel('Number of Patients')

plt.savefig('./overview_data/Resistance_distribution.png')

#check for missing values
print(f"Missing values in the features: {data_feats.isnull().sum().sum()}")
print(f"Missing values in the outcome variable: {data_labels.isnull().sum().sum()}")

#creating a new dataframe that includes features and labels
new_column = data_labels['label']
data_feats_with_label = pd.concat([data_feats, new_column], axis=1)
print(data_feats_with_label.head(5))
print(data_feats_with_label.dtypes.unique())

#distribution of some random features
random.seed(42)
random_feats = random.sample(data_feats.columns.tolist(), 10) #list of 10 random features
print(random_feats)

fig, ax = plt.subplots(2, 5, figsize=(12, 6))

for i, ax in enumerate(ax.flatten()):
    
    sns.histplot(data=data_feats_with_label, 
                 x= random_feats[i], hue='label', 
                 bins=50, 
                 palette='viridis', 
                 ax=ax
    )
    ax.set_xlabel(f'Feature Value {random_feats[i]}')
    ax.set_ylabel('Number of Patients')
    
plt.suptitle('Distribution of some random chosen Features')
fig.tight_layout()

plt.savefig('./overview_data/Random_Features_distribution.png')

###test for normality in the random features###
# extract the two groups you want to compare
for feat in random_feats:
    group_1 = data_feats_with_label.loc[data_feats_with_label['label'] == 0, feat]
    group_2 = data_feats_with_label.loc[data_feats_with_label['label'] == 1, feat]
    
    # normality test: if p-vlaue < 0.05 -> data is not normally distributed
    pval_1 = sts.shapiro(group_1).pvalue 
    pval_2 = sts.shapiro(group_2).pvalue
    print(f"Shapiro-Wilk test for {feat}:")
    print(f"Non-resistant Group: {pval_1}")
    print(f"Resistant Group: {pval_2}")

    #statistical test for the two groups
    statistical_test = sts.ranksums(group_1, group_2)
    print(f"P-value of Statistical test for {feat} between the two labels: {statistical_test.pvalue}")

    significance_level = 0.05 / len(random_feats)
    #if p-value < significance level -> the two groups(resistant not resistant) are significantly different in that feature
    if statistical_test.pvalue < significance_level:
        print(f"Feature {feat} is significantly different between the two groups")
    else:
        print(f"Feature {feat} is not significantly different between the two groups")

#probably most features like these random features are not normally disributed
#some fetaures seem to be significantly different between the two groups, some don't

