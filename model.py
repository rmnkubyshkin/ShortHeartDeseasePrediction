import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, validation_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import pickle


df = pd.DataFrame([[1, 2, 3], [3, 4, 6]])

data = pd.read_csv("D:\Personal Projects\Programing\pyth pr\Short Heart Desease Project\heart_desease_data.csv")
data.info()


#cp, thalach, exang, oldpeak, slope, ca

data.info()

data.groupby('target')

data.describe()

#plt.subplots(figsize=(20, 15))
#sns.heatmap(data.corr(), annot=True)

excluded_features = ["age", "sex", "trestbps", "chol", "fbs", "restecg", "thal", "target"]
features = data.drop(excluded_features, axis=1)
label = data["target"]

print(features)

X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.4, random_state=42)

X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

for data in [y_train, y_val, y_test]:
    print(round(len(data) / len(label), 2))

rf = RandomForestClassifier()

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

accuracy = round(accuracy_score(y_test, y_pred), 3)
precision = round(precision_score(y_test, y_pred, average="weighted"), 3)
recall = round(recall_score(y_test, y_pred, average="weighted"), 3)
print("Max Depth: {} || Estimators: {} || Acuracy: {} || Precision: {} || Recall: {}".
      format(rf.max_depth, rf.n_estimators, accuracy, precision, recall))

scores = []
for k in range(1, 10):
    rfc = RandomForestClassifier(n_estimators=k)
    rfc.fit(X_train, y_train)
    y_pred = rfc.predict(X_test)
    scores.append(accuracy_score(y_test, y_pred))

scores_2 = []
for j in range(1, 30):
    rfc = RandomForestClassifier(n_estimators=5, max_depth=j)
    rfc.fit(X_train, y_train)
    y_pred = rfc.predict(X_test)
    scores_2.append(accuracy_score(y_test, y_pred))

#plt.figure(figsize=(10, 5))
#plt.plot(range(1, 30), scores_2)
#plt.xlabel('Value of n_estimators for Random Forest Classifier')
#plt.ylabel('Testing Accuracy')

score = cross_val_score(rf, X_train, y_train.values.ravel(), cv=2)
score.mean()

hyperparams = {
    'n_estimators': [5, 10, 15, 25, 40, 50],
    'max_depth': [3, 5, 10, 15, 25, None]
}
cross_val = GridSearchCV(rf, hyperparams, cv=2)
cross_val.fit(X_train, y_train.values.ravel())


def results(results):
    print("Optimal Hyperparams: {}\n".format(results.best_params_))
    means = results.cv_results_['mean_test_score']
    stds = results.cv_results_['std_test_score']

    for mean, std, params in zip(means, stds, results.cv_results_['params']):
        print("Mean {} standart Deviation {} Hyperparameters {}"
              .format(round(mean, 3), round(std * 2, 2), params))


results(cross_val)

rf1 = RandomForestClassifier(n_estimators=25, max_depth=15)
rf1.fit(X_train, y_train.values.ravel())
rf2 = RandomForestClassifier(n_estimators=25, max_depth=3)
rf2.fit(X_train, y_train.values.ravel())
rf3 = RandomForestClassifier(n_estimators=5, max_depth=5)
rf3.fit(X_train, y_train.values.ravel())

for md1 in [rf1, rf2, rf3]:
    y_pred = md1.predict(X_val)
    accuracy = round(accuracy_score(y_val, y_pred), 3)
    precision = round(precision_score(y_val, y_pred, average="weighted"), 3)
    recall = round(recall_score(y_val, y_pred, average="weighted"), 3)

    print("Max Depth: {} || Estimators: {} || Acuracy: {} || Precision: {} || Recall: {}".
          format(md1.max_depth, md1.n_estimators, accuracy, precision, recall))

y_pred = rf2.predict(X_test)
accuracy = round(accuracy_score(y_test, y_pred), 3)
precision = round(precision_score(y_test, y_pred, average="weighted"), 3)
recall = round(recall_score(y_test, y_pred, average="weighted"), 3)
print('\n' "Max Depth: {} || Estimators: {} || Acuracy: {} || Precision: {} || Recall: {}".
      format(rf2.max_depth, rf2.n_estimators, accuracy, precision, recall))


pickle.dump(rf1, open('model.pkl', 'wb'))

model = pickle.load(open('model.pkl', 'rb'))
