import pandas as pd
import numpy as np

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Evaluation
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# -----------------------------------
# LOAD DATASET
# -----------------------------------

df = pd.read_csv("Crop_recommendation.csv")

print("Dataset Shape:", df.shape)

# DISPLAY SOME DATASET ROWS
print("\nFirst 5 Rows of Dataset:")
print(df.head())


# -----------------------------------
# ADD SEASON COLUMN
# -----------------------------------
# 0 = Summer
# 1 = Winter
# 2 = Monsoon

np.random.seed(42)

df["season"] = np.random.randint(0, 3, len(df))


# -----------------------------------
# FEATURES AND TARGET
# -----------------------------------

X = df.drop("label", axis=1)

y = df["label"]

encoder = LabelEncoder()

y = encoder.fit_transform(y)


# -----------------------------------
# TRAIN TEST SPLIT
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# -----------------------------------
# FEATURE SCALING
# -----------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)


# ===================================
# NEW INPUT DATA
# ===================================

new_data = [[90, 42, 43, 21, 82, 6.5, 200, 2]]

new_data = scaler.transform(new_data)


# ===================================
# 1 LOGISTIC REGRESSION
# ===================================

print("\n==============================")
print("LOGISTIC REGRESSION")
print("==============================")

lr = LogisticRegression(max_iter=1000)

lr.fit(X_train, y_train)

pred = lr.predict(X_test)

print("Accuracy:",
      accuracy_score(y_test, pred))

# SMALL CONFUSION MATRIX

cm = confusion_matrix(y_test, pred)

print("\nConfusion Matrix Shape:", cm.shape)

print(cm[:5, :5])   # only first 5 rows and columns


# SMALL CLASSIFICATION REPORT

report = classification_report(
    y_test,
    pred,
    target_names=encoder.classes_,
    output_dict=True
)

report_df = pd.DataFrame(report).transpose()

print("\nClassification Report (First 5 Classes):")

print(report_df.head())


crop = lr.predict(new_data)

print("\nPredicted Crop:",
      encoder.inverse_transform(crop)[0])


# ===================================
# 2 DECISION TREE
# ===================================

print("\n==============================")
print("DECISION TREE")
print("==============================")

dt = DecisionTreeClassifier(random_state=42)

dt.fit(X_train, y_train)

pred = dt.predict(X_test)

print("Accuracy:",
      accuracy_score(y_test, pred))

cm = confusion_matrix(y_test, pred)

print("\nConfusion Matrix Shape:", cm.shape)

print(cm[:5, :5])

report = classification_report(
    y_test,
    pred,
    target_names=encoder.classes_,
    output_dict=True
)

report_df = pd.DataFrame(report).transpose()

print("\nClassification Report (First 5 Classes):")

print(report_df.head())

crop = dt.predict(new_data)

print("\nPredicted Crop:",
      encoder.inverse_transform(crop)[0])


# ===================================
# 3 RANDOM FOREST
# ===================================

print("\n==============================")
print("RANDOM FOREST")
print("==============================")

rf = RandomForestClassifier(
    n_estimators=50,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

pred = rf.predict(X_test)

print("Accuracy:",
      accuracy_score(y_test, pred))

cm = confusion_matrix(y_test, pred)

print("\nConfusion Matrix Shape:", cm.shape)

print(cm[:5, :5])

report = classification_report(
    y_test,
    pred,
    target_names=encoder.classes_,
    output_dict=True
)

report_df = pd.DataFrame(report).transpose()

print("\nClassification Report (First 5 Classes):")

print(report_df.head())

crop = rf.predict(new_data)

print("\nPredicted Crop:",
      encoder.inverse_transform(crop)[0])


# ===================================
# 4 KNN
# ===================================

print("\n==============================")
print("KNN")
print("==============================")

knn = KNeighborsClassifier()

knn.fit(X_train, y_train)

pred = knn.predict(X_test)

print("Accuracy:",
      accuracy_score(y_test, pred))

cm = confusion_matrix(y_test, pred)

print("\nConfusion Matrix Shape:", cm.shape)

print(cm[:5, :5])

report = classification_report(
    y_test,
    pred,
    target_names=encoder.classes_,
    output_dict=True
)

report_df = pd.DataFrame(report).transpose()

print("\nClassification Report (First 5 Classes):")

print(report_df.head())

crop = knn.predict(new_data)

print("\nPredicted Crop:",
      encoder.inverse_transform(crop)[0])


# ===================================
# 5 NAIVE BAYES
# ===================================

print("\n==============================")
print("NAIVE BAYES")
print("==============================")

nb = GaussianNB()

nb.fit(X_train, y_train)

pred = nb.predict(X_test)

print("Accuracy:",
      accuracy_score(y_test, pred))

cm = confusion_matrix(y_test, pred)

print("\nConfusion Matrix Shape:", cm.shape)

print(cm[:5, :5])

report = classification_report(
    y_test,
    pred,
    target_names=encoder.classes_,
    output_dict=True
)

report_df = pd.DataFrame(report).transpose()

print("\nClassification Report (First 5 Classes):")

print(report_df.head())

crop = nb.predict(new_data)

print("\nPredicted Crop:",
      encoder.inverse_transform(crop)[0])


# ===================================
# USER INPUT DATA
# ===================================

N = float(input("\nEnter Nitrogen value: "))
P = float(input("Enter Phosphorus value: "))
K = float(input("Enter Potassium value: "))

temperature = float(input("Enter Temperature: "))
humidity = float(input("Enter Humidity: "))
ph = float(input("Enter pH value: "))
rainfall = float(input("Enter Rainfall: "))

print("\nSeason:")
print("0 = Summer")
print("1 = Winter")
print("2 = Monsoon")

season = int(input("Enter Season: "))

user_data = [[
    N,
    P,
    K,
    temperature,
    humidity,
    ph,
    rainfall,
    season
]]

user_data = scaler.transform(user_data)

user_crop = lr.predict(user_data)

print("\nUser Predicted Crop:",
      encoder.inverse_transform(user_crop)[0])

#user input
N = float(input("Nitrogen: "))
P = float(input("Phosphorus: "))
K = float(input("Potassium: "))

temperature = float(input("Temperature: "))

humidity = float(input("Humidity: "))

ph = float(input("pH: "))

rainfall = float(input("Rainfall: "))

print("0 Summer")
print("1 Winter")
print("2 Monsoon")

season = int(input("Season: "))

user = pd.DataFrame(
    [[
        N,
        P,
        K,
        temperature,
        humidity,
        ph,
        rainfall,
        season
    ]],
    columns=X.columns
)

user = scaler.transform(user)

prediction = rf.predict(user)

print(
    "\nRecommended Crop:"
)

print(
    encoder.inverse_transform(
        prediction
    )[0]
)