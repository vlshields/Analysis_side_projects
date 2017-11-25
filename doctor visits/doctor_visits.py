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
from datetime import datetime

#read the data
doctors = pd.read_csv('doctor.csv')

def percent_show_up(series):
    count = 0
    for i in series:
        if i == 'No':
            count+= 1
    print(count,'patients actually show up in the dataset')
 
    print('percent show up: {}'.format(count / doctors.shape[0]))

def clean_dates_T(column):
    return column.replace("T",":")

def clean_dates_Z(column):
    return column.replace("Z","")

def get_date(column):
    new_time = datetime.strptime(column, '%Y-%m-%d:%H:%M:%S')
    return new_time

def get_hour(date):
    new_date = date.hour
    return new_date


# Clean dates
doctors['ScheduledDay'] = doctors['ScheduledDay'].apply(clean_dates_T)
doctors['ScheduledDay'] = doctors['ScheduledDay'].apply(clean_dates_Z)
doctors['new_date'] = doctors['ScheduledDay'].apply(get_date)
doctors['hour_scheduled'] = doctors['new_date'].apply(get_hour)


def split_data(df):
    """Separate the data"""
    
    y = df['No-show']

    X = df[['Age', 'Scholarship', 'Hipertension', 'Diabetes', 'Alcoholism', 'Gender', 'Handcap', 
                            'SMS_received','hour_scheduled']]
    return X,y                        

X, y = split_data(doctors)

# check nulls
print (y.isnull().sum())

for feature in X:
    print("There are",X[feature].isnull().sum(),"missing values in",feature)

# convert dummies to 0 and 1
d = {'M':0, 'F':1}
X['Gender'] = X['Gender'].apply(lambda x: d[x])


# Same thing for the classification. 0 means they did show up and 1 means they did not.
d = {'No':0, 'Yes':1}
y = y.apply(lambda x: d[x])

def arrays(x,y):
    """change to numpy arrays"""
    
    x = np.array(x)
    y = np.array(y)
    return x,y

X,y = arrays(X,y)

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

# compare this to the test set accuracy
percent_show_up(doctors['No-show'])

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
