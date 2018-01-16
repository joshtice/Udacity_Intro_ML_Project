#!/usr/bin/python

# General imports
from __future__ import division
import numpy as np
import pickle
import sys

# Scikit-Learn imports
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import chi2
from sklearn.svm import SVC

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

# Task 1: Select what features you'll use.
# features_list is a list of strings, each of which is a feature name.
# The first feature must be "poi".

features_list = [
    'poi',  # Must be listed first
    'bonus',
    'deferral_payments',
    'deferred_income',
    'director_fees',
    #'email_address',
    'exercised_stock_options',
    'expenses',
    'from_messages',
    'from_poi_to_this_person',
    'from_this_person_to_poi',
    'loan_advances',
    'long_term_incentive',
    'other',
    'restricted_stock',
    'restricted_stock_deferred',
    'salary',
    'shared_receipt_with_poi',
    'to_messages',
    'total_payments',
    'total_stock_value'
]

# Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# Task 2: Remove outliers
del(data_dict['TOTAL'])

# Task 3: Create new feature(s)
# Store to my_dataset for easy export below.
for key in data_dict.keys():
    if (data_dict[key]['from_poi_to_this_person'] == 'NaN') or \
       (data_dict[key]['from_messages'] == 'NaN'):
        data_dict[key]['ratio_poi_from_messages'] = 'NaN'
    else:
        data_dict[key]['ratio_poi_from_messages'] = \
            data_dict[key]['from_poi_to_this_person'] / \
            data_dict[key]['from_messages']
    if (data_dict[key]['from_this_person_to_poi'] == 'NaN') or \
       (data_dict[key]['to_messages'] == 'NaN'):
        data_dict[key]['ratio_poi_to_messages'] = 'NaN'
    else:
        data_dict[key]['ratio_poi_to_messages'] = \
            data_dict[key]['from_this_person_to_poi'] / \
            data_dict[key]['to_messages']
features_list.append('ratio_poi_from_messages')
features_list.append('ratio_poi_to_messages')

my_dataset = data_dict

# Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

# Task 4: Try a varity of classifiers
# Please name your classifier clf for easy export below.
# Note that if you want to do PCA or other multi-stage operations,
# you'll need to use Pipelines. For more info:
# http://scikit-learn.org/stable/modules/pipeline.html

# Task 5: Tune your classifier to achieve better than .3 precision and recall
# using our testing script. Check the tester.py script in the final project
# folder for details on the evaluation method, especially the test_classifier
# function. Because of the small size of the dataset, the script uses
# stratified shuffle split cross validation. For more info:
# http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

clf = DecisionTreeClassifier(
    criterion='gini',
    splitter='best',
    max_depth=3,
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_features=4,
    random_state=42,
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    min_impurity_split=None,
    class_weight='balanced',
    presort=False
)

# Task 6: Dump your classifier, dataset, and features_list so anyone can
# check your results. You do not need to change anything below, but make sure
# that the version of poi_id.py that you submit can be run on its own and
# generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
