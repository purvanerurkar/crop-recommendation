import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ======================
# MODELS
# ======================

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# ======================
# PREPROCESSING
# ======================

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler,
    label_binarize
)

# ======================
# METRICS
# ======================

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

from sklearn.multiclass import OneVsRestClassifier


# =========================
# LOAD DATASET
# =========================

df = pd.read_csv("Crop_recommendation.csv")

print("Dataset Shape")
print(df.shape)

print("\nFirst 5 Rows")
print(df.head())

print("\nDataset Info")
print(df.info())

print("\nDescription")
print(df.describe())

print("\nMissing Values")
print(df.isnull().sum())


# =========================
# ADD SEASON COLUMN
# =========================

np.random.seed(42)

df["season"] = np.random.randint(
    0,
    3,
    len(df)
)


# =========================
# FEATURES & TARGET
# =========================

X = df.drop("label", axis=1)

y = df["label"]

encoder = LabelEncoder()

y = encoder.fit_transform(y)


# =========================
# SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,
    test_size=0.2,
    random_state=42

)


# =========================
# SCALING
# =========================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)


new_data = pd.DataFrame(
[[90,42,43,21,82,6.5,200,2]],
columns=X.columns
)

new_data = scaler.transform(new_data)


accuracies=[]
models=[]


# =========================
# ROC FUNCTION
# =========================

def plot_roc(model,name):

    try:

        y_test_bin = label_binarize(
            y_test,
            classes=np.unique(y)
        )

        clf = OneVsRestClassifier(model)

        clf.fit(
            X_train,
            y_train
        )

        y_score = clf.predict_proba(
            X_test
        )

        plt.figure(figsize=(8,6))

        for i in range(
            len(np.unique(y))
        ):

            fpr,tpr,_ = roc_curve(

                y_test_bin[:,i],

                y_score[:,i]

            )

            roc_auc = auc(
                fpr,
                tpr
            )

            plt.plot(
                fpr,
                tpr,
                label=f"AUC={roc_auc:.2f}"
            )

        plt.title(name+" ROC")

        plt.legend()

        plt.show()

        plt.close()

    except:

        print(
        name,
        "ROC not supported"
        )


################################################
# LOGISTIC REGRESSION
################################################

print("\nLOGISTIC REGRESSION")

lr = LogisticRegression(max_iter=1000)

lr.fit(X_train,y_train)

pred = lr.predict(X_test)

lr_acc = accuracy_score(
    y_test,
    pred
)

print(
"Accuracy:",
lr_acc
)

accuracies.append(lr_acc)

models.append("LR")

print(

classification_report(

y_test,
pred,

target_names=
encoder.classes_

)

)

cm = confusion_matrix(
y_test,
pred
)

plt.figure(figsize=(10,8))

sns.heatmap(
cm,
annot=True,
fmt="d"
)

plt.title(
"LR Confusion Matrix"
)

plt.show()

plt.close()

plot_roc(
LogisticRegression(max_iter=1000),
"LR"
)

crop = lr.predict(new_data)

print(
"Crop:",
encoder.inverse_transform(crop)[0]
)



################################################
# DECISION TREE
################################################

print("\nDECISION TREE")

dt=DecisionTreeClassifier()

dt.fit(
X_train,
y_train
)

pred=dt.predict(X_test)

dt_acc=accuracy_score(
y_test,
pred
)

print(
"Accuracy:",
dt_acc
)


print(

classification_report(

y_test,
pred,

target_names=
encoder.classes_

)

)

cm=confusion_matrix(
y_test,
pred
)

plt.figure(figsize=(10,8))

sns.heatmap(
cm,
annot=True,
fmt="d"
)
plt.title(
"DT confusion Matrix"
)

plt.show()

plt.close()

plot_roc(
DecisionTreeClassifier(),
"DT"
)

################################################
# KNN
################################################

print("\nKNN")

knn=KNeighborsClassifier()

knn.fit(
X_train,
y_train
)

pred=knn.predict(X_test)

knn_acc=accuracy_score(
y_test,
pred
)

print(
"Accuracy:",
knn_acc
)


print(classification_report(y_test,pred,target_names=encoder.classes_))

cm=confusion_matrix(y_test,pred)

plt.figure(figsize=(10,8))

sns.heatmap(
cm,
annot=True,
fmt="d"
)
plt.title(
"KNN confusion Matrix"
)

plt.show()

plt.close()

plot_roc(
KNeighborsClassifier(),
"KNN"
)


################################################
# NAIVE BAYES
################################################

print("\nNAIVE BAYES")

nb=GaussianNB()

nb.fit(
X_train,
y_train
)

pred=nb.predict(X_test)

nb_acc=accuracy_score(
y_test,
pred
)

print(
"Accuracy:",
nb_acc
)

print(

classification_report(

y_test,
pred,

target_names=
encoder.classes_

)

)

cm=confusion_matrix(
y_test,
pred
)

plt.figure(figsize=(10,8))

sns.heatmap(
cm,
annot=True,
fmt="d"
)
plt.title(
"Naive Bayes confusion Matrix"
)

plt.show()

plt.close()

plot_roc(
GaussianNB(),
"NB"
)


