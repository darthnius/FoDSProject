### FoDS project 3: DRIAMS-EC ###

#import modules
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

#load data
data_features = pd.read_csv("../data/driams_Escherichia coli_Ceftriaxone_features.csv")
data_labels = pd.read_csv("../data/driams_Escherichia coli_Ceftriaxone_labels.csv")
data_features = data_features.rename(columns={data_features.columns[0]: "ID"})
data_labels = data_labels.rename(columns={data_labels.columns[0]: "ID"})
data = pd.merge(data_features, data_labels, on="ID", how="inner")
print(data.head(3))

#data exploring
n_rows = data.shape[0] 
n_cols = data.shape[1]
print(f"The data contains {n_rows} rows and {n_cols} columns.")

#missing data
missing_data = data.isnull().any()
columns_with_missing_data = missing_data[missing_data].index.tolist()
print(missing_data)
print(f"Missing data found in {len(columns_with_missing_data)} columns.")

###### Support Vector Maschine Model #######
data = data.drop(columns=['ID'])
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle class imbalance using SMOTE on the training set only
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

### ANOVA for SVM ###
# Define a function to evaluate different numbers of features
def evaluate_features(X_train, y_train, k):
    selector = SelectKBest(f_classif, k=k)
    scaler = StandardScaler()
    svm_model = SVC(kernel='rbf', C=1, gamma='scale')
    
    pipeline = Pipeline([
        ('feature_selection', selector),
        ('scaler', scaler),
        ('svm', svm_model)
    ])
    
    # Evaluate using cross-validation
    scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
    return scores.mean()

# Test different numbers of features
num_features = [10, 50, 100, 150, 200, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 3000, 4000, 5000, 6000]
results = {k: evaluate_features(X_train_resampled, y_train_resampled, k) for k in num_features}

# Find the best number of features
best_k = max(results, key=results.get)
print(f"Best number of features (SVM): {best_k}")
print(f"Accuracy with {best_k} features (SVM): {results[best_k]}")

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(num_features, [results[k] for k in num_features], marker='o')
plt.xlabel('Number of Features')
plt.ylabel('Cross-Validated Accuracy')
plt.title('Feature Selection using ANOVA for SVM')
plt.grid()
plt.savefig('../output/feature_selection_svm.png')
plt.show()

# Now use the best number of features to train the final model
best_k = 1500 # search plateau in the ANOVA plot
selector = SelectKBest(f_classif, k=best_k)
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

# Define a parameter grid to search for the best parameters
param_grid = {
  'C': [0.1, 1, 10, 100],
  'gamma': [0.001, 0.01, 0.1, 1],
  'class_weight': [{0: 1, 1: 0.5}, {0: 1, 1: 0.75}, {0: 1, 1: 1}, {0: 1, 1: 0.25}]
}

# GridSearchCV to find the best parameters
grid_search = GridSearchCV(SVC(kernel='rbf', probability=True), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_resampled_new, y_train_resampled)

# Print the best parameters and best score
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation accuracy:", grid_search.best_score_)

# Train the final model with the best parameters
best_model = grid_search.best_estimator_
best_model.fit(X_train_resampled_new, y_train_resampled)

# Evaluation
y_pred = best_model.predict(X_test_new)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, output_dict=True)
print(conf_matrix)
print(class_report)

# Convert confusion matrix to a DataFrame
conf_matrix_df = pd.DataFrame(conf_matrix, index=['Actual_0', 'Actual_1'], columns=['Predicted_0', 'Predicted_1'])
conf_matrix_df.to_csv('../output/confusion_matrix_svm.csv')

# Convert classification report to a DataFrame
class_report_df = pd.DataFrame(class_report).transpose()
class_report_df.to_csv('../output/classification_report_svm.csv')

# Predict probabilities for the test set
y_prob = best_model.predict_proba(X_test_new)[:, 1]  # Get the probabilities for the positive class

# Compute ROC curve and ROC area
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")

# Save the ROC curve figure
plt.savefig('../output/roc_curve.png')
plt.show()