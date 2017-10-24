import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.tree import export_graphviz
from sklearn.metrics import accuracy_score
from sklearn import tree
from functools import partial

# balance_data = pd.read_csv('../data/balance-scale.data',
                           # sep= ',', header= None)

# print "Dataset Lenght:: ", len(balance_data)
# print "Dataset Shape:: ", balance_data.shape
#
# print "Dataset:: "
# print balance_data.head()

# X = balance_data.values[:, 1:5]
# Y = balance_data.values[:,0]

# X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)
# print X_train
# print y_train
# for (x,y), value in np.ndenumerate(X_train):
#     print x,y

frame = pd.read_csv('../data/mtcars.csv')
X_train = frame[['wt']]
y_train = frame[['am']]
frame = frame[['wt', 'am']]
min_samples_leaf = 2
clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = None,
                                        max_depth=None, min_samples_leaf=min_samples_leaf,
                                        min_samples_split = 3*min_samples_leaf)
clf = clf_gini.fit(X_train, y_train)
print clf

# clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,
#          max_depth=3, min_samples_leaf=5)
# clf = clf_entropy.fit(X_train, y_train)


with open("dtree2.dot", 'w') as dotfile:
    export_graphviz(clf, out_file = dotfile)

# feature = clf.tree_.feature
feature_names = ['wt']
# tree_ = clf.tree_
# feature_name = [
#                 feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
#                         for i in tree_.feature
#                             ]
# print "def tree({}):".format(", ".join(feature_names))

# frame['labels'] = len(frame['wt'])*0
# print frame

def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
                    feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
                    for i in tree_.feature
                    ]
    print "def tree({}):".format(", ".join(feature_names))

    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            print "{}if {} <= {}:".format(indent, name, threshold)
            recurse(tree_.children_left[node], depth + 1)
            print "{}else:  # if {} > {}".format(indent, name, threshold)
            recurse(tree_.children_right[node], depth + 1)
        else:
            print "{}return {}".format(indent, tree_.value[node])

    recurse(0, 1)


tree_to_code(clf, feature_names)


class DT(object):
    def __init__(self, tree, feature_names):
        self.tree_ = tree.tree_
        self.feature_names = feature_names
        self.thresholds = []
        self.find_threshold(0, 1)
        self.thresholds = self.sort_thresholds(self.thresholds)

    def find_threshold(self, node, depth):
        if self.tree_.feature[node] != _tree.TREE_UNDEFINED:
            threshold = self.tree_.threshold[node]
            self.thresholds.append(threshold)
            self.find_threshold(self.tree_.children_left[node], depth + 1)
            self.thresholds.append(threshold)
            self.find_threshold(self.tree_.children_right[node], depth + 1)

    def sort_thresholds(self, thresholds):
        thresholds = list(set(thresholds))
        thresholds.sort()
        return thresholds

clf = DT(clf, feature_names)
print clf.thresholds

print dir(clf.tree_)
print clf.tree_.impurity

bins = np.append((-float('inf'),), clf.thresholds)
bins = np.append(bins, (float('inf'),))
print bins

bin_dict = {'lower': np.append((-float('inf'),), clf.thresholds),
            'upper': np.append(clf.thresholds, (float('inf'),)),
           }
bin_dict['labels'] = np.arange(0, len(bin_dict['lower']))
print bin_dict

def create_label(row):
    # for threshold in bins:
    #     if row['wt'] > threshold:
    #         label = threshold
    # label = label if bin_dict['lower'][ind]<=row['wt']<bin_dict['upper'][ind]\
    label = None
    for ind, label in enumerate(bin_dict['labels']):
        if bin_dict['lower'][ind]<=row['wt']<bin_dict['upper'][ind]:
            label = label
            break
                  # else None
    return label

frame['label'] = frame.apply(create_label, axis=1)
print frame
