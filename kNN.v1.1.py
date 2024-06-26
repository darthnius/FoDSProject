import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.impute import SimpleImputer



data = pd.read_csv("C:/Users/nilss/Documents/GitHub/driams_Escherichia coli_Ceftriaxone_features.csv")

label = pd.read_csv("C:/Users/nilss/Documents/GitHub/driams_Escherichia coli_Ceftriaxone_labels.csv")

X = data[data.columns[1:]]
y = label[label.columns[1:]]


imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit(X)
X_imp= imp.transform(X)

pca = {}
pca_X_train = {}
pca_X_test = {}
knn = {}
yhat = {}
accuracy = {}
precision = {}
recall = {}
folds = 10
base = 400
rate = 50
split = 0.3


X_train, X_test, y_train, y_test = train_test_split(X_imp, y, test_size=split, random_state=69)

std_scaler = StandardScaler()
scaled_X_train = std_scaler.fit_transform(X_train)
scaled_X_test  = std_scaler.transform(X_test)



fig, ax = plt.subplots(figsize=(12, 12))
for i in range(1,folds):
  pca[i] = PCA(n_components=base+i*rate)
  pca[i].fit_transform(scaled_X_train)
  print(sum(pca[i].explained_variance_ratio_))
  print(pca[i].n_components_)
  pca_X_train[i] = pca[i].transform(scaled_X_train) 
  pca_X_test[i] = pca[i].transform(scaled_X_test)
  knn[i] = KNeighborsClassifier()
  knn[i].fit(pca_X_train[i], y_train.values.ravel())
  fpr, tpr, thresholds = metrics.roc_curve(y_test, knn[i].predict_proba(pca_X_test[i])[:,1])
  auc = metrics.roc_auc_score(y_test,knn[i].predict(pca_X_test[i]))
  plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % (base+i*rate, auc))
  yhat[i] = knn[i].predict(pca_X_test[i])
  accuracy[i] = accuracy_score(y_test, yhat[i])
  precision[i] = precision_score(y_test, yhat[i])
  recall[i] = recall_score(y_test, yhat[i])

plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC-Curve")
plt.legend(loc="lower right")
plt.savefig(f"C:/Users/nilss/Documents/GitHub/FoDSProject/ROC_{str(split)}_split_{str(base)}_base{str(rate)}_rate.pdf")

for j in range(1,folds):
  print("Fold : ", base+j*rate)
  print("Accuracy:", accuracy[j])
  print("Precision:", precision[j])
  print("Recall:", recall[j])




