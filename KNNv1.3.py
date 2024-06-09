import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_recall_fscore_support as  score
from sklearn import metrics
# Define a function to evaluate different numbers of features
def evaluate_features_knn(k, n_neighbors):
    selector = SelectKBest(f_classif, k=k)
    scaler = StandardScaler()
    knn_model = KNeighborsClassifier(n_neighbors=n_neighbors)
    
    pipeline = Pipeline([
        ('feature_selection', selector),
        ('scaler', scaler),
        ('knn', knn_model)
    ])
    
    # Evaluate using cross-validation
    scores = cross_val_score(pipeline, X_train_resampled, y_train_resampled, cv=5, scoring='accuracy')
    return scores.mean()

#variables
num_Neighbor = 11
split = 0.2

#read data
data_features = pd.read_csv("../data/driams_Escherichia coli_Ceftriaxone_features.csv")
data_labels = pd.read_csv("../data/driams_Escherichia coli_Ceftriaxone_labels.csv")
data_features = data_features.rename(columns={data_features.columns[0]: "ID"})
data_labels = data_labels.rename(columns={data_labels.columns[0]: "ID"})
data = pd.merge(data_features, data_labels, on="ID", how="inner")
data = data.drop(columns=['ID'])
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

error_rates = []
for i in np.arange(1, 302, step=2):
    new_model = KNeighborsClassifier(n_neighbors = i)
    new_model.fit(X_train_resampled, y_train_resampled)
    new_predictions = new_model.predict(X_test)
    print(np.mean(new_predictions != y_test))
    error_rates.append(np.mean(new_predictions != y_test))

plt.figure(figsize=(16,12))
plt.xlabel('Number of Neighbors')
plt.ylabel('Error Rate')
plt.title('Elbow Plot for n Selection')
plt.plot(error_rates)
plt.savefig(f"../FoDSProject/kNN_performance/kNN_Elbow_{str(split)}_split.png")
plt.close()


num_features = np.arange(1, 300)
results_knn = {k: evaluate_features_knn(k,num_Neighbor) for k in num_features}

# Find the best number of features
best_k_knn = max(results_knn, key=results_knn.get)
print(f"Best number of features (KNN): {best_k_knn}")
print(f"Accuracy with {best_k_knn} features (KNN): {results_knn[best_k_knn]}")

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(num_features, [results_knn[k] for k in num_features], marker='o')
plt.xlabel('Number of Features')
plt.ylabel('Cross-Validated Accuracy')
plt.title('Feature Selection using ANOVA for KNN')
plt.grid()
plt.savefig(f"../FoDSProject/kNN_performance/kNN_feature_selection_{str(split)}_split{num_Neighbor}_neighbors.png")
plt.close()

selector = SelectKBest(f_classif, k=best_k_knn)
X_train_resampled_new = selector.fit_transform(X_train_resampled, y_train_resampled)
X_test_new = selector.transform(X_test)

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
X_train_resampled_new = scaler.fit_transform(X_train_resampled_new)
X_test_new = scaler.transform(X_test_new)

knn = KNeighborsClassifier(n_neighbors=num_Neighbor)
knn.fit(X_train_resampled_new, y_train)
fpr, tpr, thresholds = metrics.roc_curve(y_test, knn.predict_proba(X_test_new)[:,1])
auc = metrics.roc_auc_score(y_test,knn.predict(X_test_new))
plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % (1, auc))
yhat = knn.predict(X_test_new)

precision, recall, fscore, support = score(y_test, yhat)

plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC-Curve")
plt.legend(loc="lower right")
plt.savefig(f"../FoDSProject/kNN_performance/kNN_ROC_{str(split)}_split{num_Neighbor}_neighbors.png")
plt.close()

metric_Df = pd.DataFrame({"precision": precision.tolist(), "recall": recall.tolist(), "fscore": fscore.tolist(), "support": support.tolist()})
metric_Df.to_csv(f"../FoDSProject/kNN_performance/kNN_metrics_{str(split)}_split{num_Neighbor}_neighbors.csv")

