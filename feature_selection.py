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

### ANOVA for Logistic Regression ###
def evaluate_features_logistic_regression(k):
    selector = SelectKBest(f_classif, k=k)
    scaler = StandardScaler()
    log_reg_model = LogisticRegression(solver='liblinear')
    
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
plt.savefig('../output/feature_selection_log_reg.png')
plt.show()

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
plt.show()

### ANOVA for KNN
# Define a function to evaluate different numbers of features
def evaluate_features_knn(k, n_neighbors=5):
    selector = SelectKBest(f_classif, k=k)
    scaler = StandardScaler()
    knn_model = KNeighborsClassifier(n_neighbors=n_neighbors)
    
    pipeline = Pipeline([
        ('feature_selection', selector),
        ('scaler', scaler),
        ('knn', knn_model)
    ])
    
    # Evaluate using cross-validation
    scores = cross_val_score(pipeline, X_resampled, y_resampled, cv=5, scoring='accuracy')
    return scores.mean()

#Results for Random Forest
results_knn = {k: evaluate_features_knn(k) for k in num_features}

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
plt.savefig('../output/feature_selection_knn.png')
plt.show()
