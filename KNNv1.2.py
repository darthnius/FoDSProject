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

def evaluate_features_knn(k, n_neighbors):
    selector = SelectKBest(f_classif, k=k)
    scaler = StandardScaler()
    knn_model = KNeighborsClassifier(n_neighbors=n_neighbors)
    
    pipeline = Pipeline([
        ('feature_selection', selector),
        ('scaler', scaler),
        ('knn', knn_model)
    ])
    scores = cross_val_score(pipeline, X_resampled, y_resampled, cv=5, scoring='accuracy')
    return scores.mean()

#variables
num_Neighbor = 21
split = 0.3
best_k_num = 52

#read data
data_features = pd.read_csv("../data/driams_Escherichia coli_Ceftriaxone_features.csv")
data_labels = pd.read_csv("../data/driams_Escherichia coli_Ceftriaxone_labels.csv")
data_features = data_features.rename(columns={data_features.columns[0]: "ID"})
data_labels = data_labels.rename(columns={data_labels.columns[0]: "ID"})
data = pd.merge(data_features, data_labels, on="ID", how="inner")
data = data.drop(columns=['ID'])
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

#Elbow plot to determine k
x_training_data, x_test_data, y_training_data, y_test_data = train_test_split(X, y, test_size = split)
error_rates = []
for i in np.arange(1, 50, step=2):
    new_model = KNeighborsClassifier(n_neighbors = i)
    new_model.fit(x_training_data, y_training_data)
    new_predictions = new_model.predict(x_test_data)
    print(np.mean(new_predictions != y_test_data))
    error_rates.append(np.mean(new_predictions != y_test_data))

plt.figure(figsize=(16,12))
plt.xlabel('Number of Neighbors')
plt.ylabel('Error Rate')
plt.title('Elbow Plot for n Selection')
plt.plot(error_rates)
plt.savefig(f"../output/kNN_Elbow_{str(split)}_split.png")
plt.close()

#resampling
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)



num_features = np.arange(1, 151, step=3)
results = {k: evaluate_features_knn(k,n_neighbors=num_Neighbor) for k in num_features}

best_k = max(results, key=results.get)
print(f"Best number of features (KNN): {best_k}")
print(f"Accuracy with {best_k} features (KNN): {results[best_k]}")

plt.figure(figsize=(10, 6))
plt.plot(num_features, [results[k] for k in num_features], marker='o')
plt.xlabel('Number of Features')
plt.ylabel('Cross-Validated Accuracy')
plt.title('Feature Selection using ANOVA for KNN')
plt.grid()
plt.savefig(f"../output/kNN_feature_selection_{str(split)}_split{num_Neighbor}_neighbors.png")
plt.close()

selector = SelectKBest(f_classif, k=best_k_num)
X_resampled_new = selector.fit_transform(X_resampled, y_resampled)
X_new = selector.transform(X)
feature_scores = selector.scores_
feature_score_dict = dict(zip(data.columns[:-1], feature_scores))
sorted_feature_score_dict = dict(sorted(feature_score_dict.items(), key=lambda item: item[1], reverse=True))
top_features = list(sorted_feature_score_dict.keys())[:best_k] 
print("Top features and their scores:")
for feature in top_features:
    print(f"Feature: {feature}, Score: {sorted_feature_score_dict[feature]}")


# Standardize the features
scaler = StandardScaler()
X_resampled_new = scaler.fit_transform(X_resampled_new)
X_new = scaler.transform(X_new)


X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=split, random_state=42)

knn = KNeighborsClassifier(n_neighbors=num_Neighbor)
knn.fit(X_train, y_train)
fpr, tpr, thresholds = metrics.roc_curve(y_test, knn.predict_proba(X_test)[:,1])
auc = metrics.roc_auc_score(y_test,knn.predict(X_test))
plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % (1, auc))
yhat = knn.predict(X_test)

precision, recall, fscore, support = score(y_test, yhat)

plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC-Curve")
plt.legend(loc="lower right")
plt.savefig(f"../output/kNN_ROC_{str(split)}_split{num_Neighbor}_neighbors.png")

metric_Df = pd.DataFrame({"precision": precision.tolist(), "recall": recall.tolist(), "fscore": fscore.tolist(), "support": support.tolist()})
metric_Df.to_csv(f"../output/kNN_metrics_{str(split)}_split{num_Neighbor}_neighbors.csv")
