#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from tester import test_classifier

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

features_list = ['poi', 'salary', 'deferral_payments', 'total_payments', 'loan_advances',
                 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value',
                 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive',
                 'restricted_stock', 'director_fees', 'to_messages', 'from_poi_to_this_person',
                 'from_messages', 'from_this_person_to_poi','shared_receipt_with_poi']  # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Import package for graph plotting to detect outliers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pprint
import operator
from pandas.tools.plotting import scatter_matrix

### Create dataframe for further exploratory data analysis
df = pd.DataFrame.from_dict(data_dict, orient='index')
df = df.replace('NaN', 0)
df = df.replace([np.inf, -np.inf], 0)
df = df.apply(lambda x: x.fillna(0), axis=0)

### There are a total of 146 data points and 21 features within the dataset
print 'Shape of dataset:',
print df.shape

### There are 18 POIs and 128 non POIs within the dataset
print 'Distribution of POIs:'
print df['poi'].value_counts()

### Create a dictionary with the number of missing values of every features within the dataset
temp = {}
for i in df:
    data_name = i
    data_count = df[df[i]==0].count()[i]
    temp[data_name] = data_count

### Print out misisng values of different features. There are a lot of missing values within the datset, with loan advance
### being the being the most extreme case having 142 missing data point.
print "Count of missing value within dataset"
pp = pprint.PrettyPrinter(indent = 4)
pp.pprint(sorted(temp.items(), key=lambda x:x[1]))

### Plot scatter plot of salary against bonus to detect outliers
plt.scatter(df['salary'], df['bonus'])
plt.xlabel('salary')
plt.ylabel('bonus')
plt.draw()
plt.show()


### Task 2: Remove outliers

### From the scatter plot and pdf file, we understand that the "total" figure aggregates all the number and is thus the outlier within the data.
### "Total" will thus be removed from the data
data_dict.pop("TOTAL", 0)

### Data point "THE TRAVEL AGENCY IN THE PARK" is obviously not an employee of enron and should be excluded from the dataset.
data_dict.pop("THE TRAVEL AGENCY IN THE PARK", 0)

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.


### I hypothesize that POIs should have strong connection between each other and will receive a higher proportion of email from other POIs.
### Create a function for new feature to check the percentage of received emails of a staff which was received from POI
def email_from_poi_percentage(from_email, total_email):
    return float(from_email/total_email)

### I hypothesize that POIs should have strong connection between each other and will receive a higher proportion of email to other POIs.
### Create a function for new feature to check the percentage of sent emails of a staff which was sent to POI
def email_to_poi_percentage(to_email, total_email):
    return float(to_email/total_email)

### I hypothesize that POIs will profit from abnormally high stock value relative to salary
### Create a function for new feature as the ratio of total stock value against salary
def stock_salary_ratio(stock_value, salary):
    return float(stock_value/salary)

### Create new features within my_dataset
import math
for i in data_dict:
    if math.isnan(float(data_dict[i]['from_poi_to_this_person'])) or math.isnan(float(data_dict[i]['from_messages'])):
        data_dict[i]["email_from_poi_percentage"] = 0
    else:
        data_dict[i]["email_from_poi_percentage"] = round(email_from_poi_percentage(float(data_dict[i]['from_poi_to_this_person']),float(data_dict[i]['from_messages'])),2)
    if math.isnan(float(data_dict[i]['from_this_person_to_poi'])) or math.isnan(float(data_dict[i]['to_messages'])):
        data_dict[i]["email_to_poi_percentage"] = 0
    else:
        data_dict[i]["email_to_poi_percentage"] = round(email_to_poi_percentage(float(data_dict[i]['from_this_person_to_poi']),float(data_dict[i]['to_messages'])),2)
    if math.isnan(float(data_dict[i]['total_payments'])) or math.isnan(float(data_dict[i]['total_stock_value'])):
        data_dict[i]["total_renumeration"] = 0
    else:
        data_dict[i]["total_renumeration"] = round(float(data_dict[i]["total_payments"]) + float(data_dict[i]["total_stock_value"]),2)

# Create a list with new features on top of the original feature list
new_features_list = features_list + ["email_from_poi_percentage", "email_to_poi_percentage", "total_renumeration"]
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

# Create a naive bayes classifier
clf_NB = GaussianNB()
clf_NB = clf_NB.fit(features_train, labels_train)
pred = clf_NB.predict(features_test)
accuracy = clf_NB.score(features_test, labels_test)

print accuracy
target_names = ["Not POI", "POI"]
print '\n Naive Bayes Classification Report'
print classification_report(y_true=labels_test, y_pred=pred, target_names=target_names)

# Create scaler, PCA and svm classifier
scaler = MinMaxScaler()

# Create a svm classifier by incorporating the scaler and PCA
pipeline = Pipeline(steps=[('scaling', scaler), ("SVC", SVC())])
clf = pipeline.fit(features_train, labels_train)
pred = clf.predict(features_test)
accuracy = clf.score(features_test, labels_test)

print accuracy
target_names = ["Not POI", "POI"]
print '\n SVC Classification Report'
print classification_report(y_true=labels_test, y_pred=pred, target_names=target_names)

# Create a decision tree classifier
clf_tree = DecisionTreeClassifier()
clf_tree = clf_tree.fit(features_train, labels_train)
pred = clf_tree.predict(features_test)
accuracy = clf_tree.score(features_test, labels_test)

print accuracy
target_names = ["Not POI", "POI"]
print '\n Decision Tree Classification Report'
print classification_report(y_true=labels_test, y_pred=pred, target_names=target_names)

# Create a random forest classifier
clf_RandomForest = RandomForestClassifier()
clf_RandomForest = clf_RandomForest.fit(features_train, labels_train)
pred = clf_RandomForest.predict(features_test)
accuracy = clf_RandomForest.score(features_test, labels_test)

print accuracy
target_names = ["Not POI", "POI"]
print '\n RandomForest Classification Report'
print classification_report(y_true=labels_test, y_pred=pred, target_names=target_names)

# Create a adaboost classifier
clf_ada = AdaBoostClassifier()
clf_ada = clf_ada.fit(features_train, labels_train)
pred = clf_ada.predict(features_test)
accuracy = clf_ada.score(features_test, labels_test)

print accuracy
target_names = ["Not POI", "POI"]
print '\n Adaboost Classification Report'
print classification_report(y_true=labels_test, y_pred=pred, target_names=target_names)

# Testing the significance of new feature set with newly engineered features by trying out naive bayes algorithm
data = featureFormat(my_dataset, new_features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

# Create a naive bayes classifier with new data set
clf_NB = GaussianNB()
clf_NB = clf_NB.fit(features_train, labels_train)
pred = clf_NB.predict(features_test)
accuracy = clf_NB.score(features_test, labels_test)

print accuracy
target_names = ["Not POI", "POI"]
print '\n Naive Bayes Classification Report with new features'
print classification_report(y_true=labels_test, y_pred=pred, target_names=target_names)


### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
### Use SelectKBest to find out the most predictive features
from sklearn.feature_selection import SelectKBest
skb = SelectKBest(k = 5)
skb.fit(features, labels)
features_selected = [new_features_list[i+1] for i in skb.get_support(indices=True)]
print 'The Features Selected by SKB:'
print features_selected

# Print out the list of selected features and their respective score
features_selected_list = []
for i, j in zip(features_selected, skb.scores_):
    features_selected_list.append([i,j])
pp.pprint(sorted((features_selected_list), key = operator.itemgetter(1), reverse=True))

# Use Pipeline, SelectKBest and gridsearchcsv to choose the best number of features to be included in order to improve the naive bayes algorithm
classifier_type = GaussianNB()
params = {'SKB__k': range(1,10)}

# Employ stratified shuffle split for validation due to the small number of data points within this dataset.
sss = StratifiedShuffleSplit(labels, 10, random_state = 42)

# Pipeline function to create steps for selectkbest and NB Classifier
pipeline = Pipeline(steps = [
    ("SKB", skb),
    ("Classifier", classifier_type)
])

# GridSearchCSV for stratified shuffle split and to determine best number of features
gs = GridSearchCV(pipeline, params, scoring = 'f1', cv = sss)
gs.fit(features, labels)
print gs.best_estimator_
clf = gs.best_estimator_

pred = clf.predict(features_test)
accuracy = clf.score(features_test, labels_test)

print accuracy
target_names = ["Not POI", "POI"]
print '\n Fine Tuned Naive Bayes Classification Report'
print classification_report(y_true=labels_test, y_pred=pred, target_names=target_names)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)

