import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

#read the data
doctors = pd.read_csv('doctor.csv')

def percent_show_up(series):
    count = 0
    for i in series:
        if i == 'No':
            count+= 1
    print(count,'patients actually show up in the dataset')
 
    print('percent show up: {}'.format(count / doctors.shape[0]))

# compare this to the test set accuracy
percent_show_up(doctors['No-show'])

# Separate the data
y = doctors['No-show']

X = doctors[['Age', 'Scholarship', 'Hipertension', 'Diabetes', 'Alcoholism', 'Gender', 'Handcap', 
                        'SMS_received']]
# Check nulls
print(y.isnull().sum())

for feature in X:
    print("There are",X[feature].isnull().sum(),"missing values in",feature)

# convert dummies to 0 and 1
d = {'M':0, 'F':1}
X['Gender'] = X['Gender'].apply(lambda x: d[x])


# Same thing for the classification. 0 means they did show up and 1 means they did not.
d = {'No':0, 'Yes':1}
y = y.apply(lambda x: d[x])

# change to numpy arrays
X = np.array(X)
y = np.array(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state = 0)

# create a pipeline to see which model performs best
pipe = Pipeline([("preprocessing", None), ('classifier', RandomForestClassifier())])

param_grid = [
    
    
    {'classifier': [RandomForestClassifier(n_estimators=5)],
    'preprocessing':[None], 'classifier__max_features': [1,2,3],
    'classifier__criterion': ('gini', 'entropy'),
    'classifier__min_samples_split': [2,4,6],
    'classifier__min_samples_leaf': [1,2,3],
    'classifier__max_depth': [None,1, 2,4]},

    {'classifier': [DecisionTreeClassifier()], 'preprocessing':[None],
    'classifier__criterion': ('gini', 'entropy'),
    'classifier__splitter': ('best', 'random'),
    'classifier__max_depth': [None,1,2,4],
    'classifier__min_samples_split': [2,4,6],
    'classifier__min_samples_leaf': [1,2,3],
    'classifier__max_features': [None, 1,2,3] }

]

clf = GridSearchCV(pipe, param_grid, cv=5)
clf.fit(X_train, y_train)
pred_grid = clf.predict(X_test)
confusion = confusion_matrix(y_test, pred_grid)

print("Best params:\n{}\n".format(clf.best_params_))
print("Best cross-validation score:{}".format(clf.best_score_))
print("Test-set score: {}".format(clf.score(X_test, y_test)))
print(classification_report(y_test, pred_grid, target_names = ["Show", "No-show"]))
print("Confusion matrix:\n{}".format(confusion))


# Graph the confusion matrix

def graph_confustion(confusion):

    """Graph the confusion matrix"""

    yg = []
    xg = [1, 2, 3, 4]
    yg.append(int(confusion[0,0]))
    yg.append(int(confusion[0,1]))
    yg.append(int(confusion[1,0]))
    yg.append(int(confusion[1,1]))
    Labels = ["True Negatives", "False Positives", "False Negatives", "True Positives"]
    plt.bar(xg, yg, color='c')
    plt.xticks(xg, Labels)
    plt.xlabel('Classification')
    plt.ylabel('Number of observations')

    plt.title('Plot of confusion matrix')
    plt.show()

graph_confustion(confusion)
