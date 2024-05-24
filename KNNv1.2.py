import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

def evaluate_features_knn(k, n_neighbors=5):
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

data_features = pd.read_csv("../data/driams_Escherichia coli_Ceftriaxone_features.csv")
data_labels = pd.read_csv("../data/driams_Escherichia coli_Ceftriaxone_labels.csv")
data_features = data_features.rename(columns={data_features.columns[0]: "ID"})
data_labels = data_labels.rename(columns={data_labels.columns[0]: "ID"})
data = pd.merge(data_features, data_labels, on="ID", how="inner")
data = data.drop(columns=['ID'])
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

num_features = [10, 20, 30, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 100]

results = {k: evaluate_features_knn(k) for k in num_features}

best_k = max(results, key=results.get)
print(f"Best number of features (KNN): {best_k}")
print(f"Accuracy with {best_k} features (KNN): {results[best_k]}")

plt.figure(figsize=(10, 6))
plt.plot(num_features, [results[k] for k in num_features], marker='o')
plt.xlabel('Number of Features')
plt.ylabel('Cross-Validated Accuracy')
plt.title('Feature Selection using ANOVA for KNN')
plt.grid()
plt.savefig('../output/feature_selection_knn_smaller.png')
plt.show()