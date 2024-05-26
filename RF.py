##### Feature Selection #####
#import modules
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support as  score


#load data
data_features = pd.read_csv("../data/driams_Escherichia coli_Ceftriaxone_features.csv")
data_labels = pd.read_csv("../data/driams_Escherichia coli_Ceftriaxone_labels.csv")
data_features = data_features.rename(columns={data_features.columns[0]: "ID"})
data_labels = data_labels.rename(columns={data_labels.columns[0]: "ID"})
data = pd.merge(data_features, data_labels, on="ID", how="inner")
data = data.drop(columns=['ID'])
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)


# Test different numbers of features
num_features = [10, 50, 100, 150, 200, 250, 500, 750, 1000, 2000, 3000, 4000, 5000, 6000]

### ANOVA for Random Forest ###
# Define a function to evaluate different numbers of features
def evaluate_features_random_forest(k):
    selector = SelectKBest(f_classif, k=k)
    scaler = StandardScaler()
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    pipeline = Pipeline([
        ('feature_selection', selector),
        ('scaler', scaler),
        ('random_forest', rf_model)
    ])
    
    # Evaluate using cross-validation
    scores = cross_val_score(pipeline, X_resampled, y_resampled, cv=5, scoring='accuracy')
    return scores.mean()

#Results for Random Forest
results_ran_for = {k: evaluate_features_random_forest(k) for k in num_features}

# Find the best number of features
best_k_ran_for = max(results_ran_for, key=results_ran_for.get)
print(f"Best number of features (Random Forest): {best_k_ran_for}")
print(f"Accuracy with {best_k_ran_for} features (Random Forest): {results_ran_for[best_k_ran_for]}")

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(num_features, [results_ran_for[k] for k in num_features], marker='o')
plt.xlabel('Number of Features')
plt.ylabel('Cross-Validated Accuracy')
plt.title('Feature Selection using ANOVA for Random Forest')
plt.grid()
plt.savefig('../output/feature_selection_ran_for.png')


### Now use the best number of features to train the final model
best_k = 500 #search plateau in the ANOVA plot!
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

#########################################################################

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

precision, recall, fscore, support = score(y_test, y_pred_rf)

#########################################################################
y_prob_rf = rf_model.predict_proba(X_test)

fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, y_prob_rf[:, 1])

auc_rf = auc(fpr_rf, tpr_rf)

# ROC-curve
plt.figure(figsize=(8, 6))
plt.plot(fpr_rf, tpr_rf, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_rf:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Random Forest Classifier')
plt.legend(loc='lower right')
plt.savefig('../output/RFroc_curve.png')

metric_Df = pd.DataFrame({"precision": precision.tolist(), "recall": recall.tolist(), "fscore": fscore.tolist(), "support": support.tolist()})
metric_Df.to_csv(f"../output/RF_metrics.csv")
