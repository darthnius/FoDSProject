import math
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import pandas as pd
import imblearn
import scipy.stats as sts
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import t, ttest_ind
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from imblearn.over_sampling import SMOTE

#load data
data_features = pd.read_csv("../data/DRIAMS-EC/driams_Escherichia coli_Ceftriaxone_features.csv")
data_labels = pd.read_csv("../data/DRIAMS-EC/driams_Escherichia coli_Ceftriaxone_labels.csv")
data_features = data_features.rename(columns={data_features.columns[0]: "ID"})
data_labels = data_labels.rename(columns={data_labels.columns[0]: "ID"})
data = pd.merge(data_features, data_labels, on="ID", how="inner")
data = data.drop(columns=['ID'])

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values


#splitting the data into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Handle class imbalance using SMOTE- creating synthetic samples for the minority class
smote = SMOTE(random_state=42) 
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

print(f'Number of features after using SMOTE: {X_resampled.shape[0]}')
print(np.unique(y_resampled, return_counts=True))


# Test different numbers of features
num_features = [10, 50, 100, 150, 200, 250, 500, 750, 1000, 2000, 3000, 4000, 5000, 6000]

### ANOVA for Logistic Regression ###
def evaluate_features_logistic_regression(k):
    selector = SelectKBest(f_classif, k=k)
    scaler = StandardScaler()
    log_reg_model = LogisticRegression(solver='liblinear', random_state=1)
    
    pipeline = Pipeline([
        ('feature_selection', selector),
        ('scaler', scaler),
        ('logistic_regression', log_reg_model)
    ])
    
    # Evaluate using cross-validation
    scores = cross_val_score(pipeline, X_resampled, y_resampled, cv=5, scoring='accuracy')
    return scores.mean()

#Results for Logistic Regression
results_log_reg = {k: evaluate_features_logistic_regression(k) for k in num_features}

# Find the best number of features
best_k_log_reg = max(results_log_reg, key=results_log_reg.get)
print(f"Best number of features (LogReg): {best_k_log_reg}")
print(f"Accuracy with {best_k_log_reg} features (LogReg): {results_log_reg[best_k_log_reg]}")

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(num_features, [results_log_reg[k] for k in num_features], marker='o')
plt.xlabel('Number of Features')
plt.ylabel('Cross-Validated Accuracy')
plt.title('Feature Selection using ANOVA for Logistic Regression')
plt.grid()
plt.savefig('./LogReg_output/feature_selection_log_reg.png')

### Now use the best number of features to train the final model
best_k = 2000 #search plateau in the ANOVA plot!
selector = SelectKBest(f_classif, k=best_k)
X_resampled_new = selector.fit_transform(X_resampled, y_resampled)
X_new = selector.transform(X)

# Best features
# Get the scores (F-values) of each feature
feature_scores = selector.scores_
# Create a dictionary mapping feature names to their scores
feature_score_dict = dict(zip(data.columns[:-1], feature_scores))
# Sort the dictionary by scores in descending order
sorted_feature_score_dict = dict(sorted(feature_score_dict.items(), key=lambda item: item[1], reverse=True))
# Print the top features and their scores
top_features = list(sorted_feature_score_dict.keys())[:10]  # Print the top 10 features
print("Top features and their scores:")
for feature in top_features:
    print(f"Feature: {feature}, Score: {sorted_feature_score_dict[feature]}")


# Standardize the features
scaler = StandardScaler()
X_resampled_new = scaler.fit_transform(X_resampled_new)
X_new = scaler.transform(X_new)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=42)


#Fit the model
logistic_regression_model = LogisticRegression(solver='liblinear', random_state=1)
logistic_regression_model.fit(X_train, y_train)

#Evaluation metrics function
def eval_Performance(y_eval, X_eval, clf, clf_name = 'My Classifier'):

    y_pred = clf.predict(X_eval)
    y_pred_proba = clf.predict_proba(X_eval)[:, 1] #probability of being positive class
    tn, fp, fn, tp = confusion_matrix(y_eval, y_pred).ravel()

    # Evaluation
    accuracy  = accuracy_score(y_eval, y_pred)
    precision = precision_score(y_eval, y_pred)
    recall    = recall_score(y_eval, y_pred)
    f1        = f1_score(y_eval, y_pred)
    fp_rates, tp_rates, _ = roc_curve(y_eval, y_pred_proba)

    # Calculate the area under the roc curve using a sklearn function
    roc_auc = auc(fp_rates, tp_rates)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fp_rates, tp_rates, label=f'ROC curve (AUC={roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color="r", ls="--", label='random\nclassifier')
    #ax.set_xlim([0.0, 1.0])
    #ax.set_ylim([0.0, 1.0])
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.legend(loc="lower right")
    ax.set_title(f'ROC curve for {clf_name}')

    plt.tight_layout()
    plt.savefig(f'./LogReg_output/ROC_curve_{clf_name}.png')

    return tp,fp,tn,fn,accuracy, precision, recall, f1, roc_auc

df_performance = pd.DataFrame(columns = ['tp','fp','tn','fn','accuracy', 'precision', 'recall', 'f1', 'roc_auc'] )

df_performance.loc['LR (test)',:] = eval_Performance(y_test, X_test, logistic_regression_model, clf_name = 'LR (test)')
df_performance.loc['LR (train)',:] = eval_Performance(y_train, X_train, logistic_regression_model, clf_name = 'LR (train)')

print(df_performance)
