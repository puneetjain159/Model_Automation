import pandas as pd
import numpy as np
import math
import copy
from sklearn.tree import DecisionTreeClassifier, _tree, DecisionTreeRegressor
from sklearn.tree import export_graphviz
from sklearn.model_selection import cross_val_score
from functools import partial

class Logger(object):
    def __init__(self):
        pass

    def info(self, msg):
        print 'INFO: {0}'.format(msg)


def _bucket_woe(x):
    t_bad = x['bad']
    t_good = x['good']
    t_bad = 0.5 if t_bad == 0 else t_bad
    t_good = 0.5 if t_good == 0 else t_good
    return np.log(t_good / t_bad)


def _calc_stat(frame, target):
    # calculating WoE
    col_names = {'count_nonzero': 'bad', 'size': 'obs'}
    stat = frame.groupby('labels')[target]\
                .agg([np.mean, np.count_nonzero, np.size])\
                .rename(columns=col_names)\
                .reset_index().copy()
    #  if self.t_type != 'b':
    stat['bad'] = stat['mean'] * stat['obs']
    stat['good'] = stat['obs'] - stat['bad']
    t_good = np.maximum(stat['good'].sum(), 0.5)
    t_bad = np.maximum(stat['bad'].sum(), 0.5)
    stat['woe'] = stat.apply(_bucket_woe, axis=1) + np.log(t_bad / t_good)
    stat['iv_stat'] = (stat['good'] / t_good - stat['bad'] / t_bad) * stat['woe']
    #  iv_stat = (stat['good'] / t_good - stat['bad'] / t_bad) * stat['woe']
    #  self.iv = iv_stat.sum()
    # adding stat data to bins
    #  self.bins = pd.merge(stat, self.bins, left_index=True, right_on=['labels'])
    #  label_woe = self.bins[['woe', 'labels']].drop_duplicates()
    frame = pd.merge(frame, stat, left_on=['labels'], right_on=['labels'])
    return stat


def create_label(row, bin_dict, value):
    label = None
    for ind, label in enumerate(bin_dict['labels']):
        if bin_dict['lower'][ind] <= row[value] < bin_dict['upper'][ind]:
            label = label
            break
    return label


class FineClass(object):
    '''will put docstring for Fineclass later'''
    def __init__(self, find_Optimum_bins=False, bins=5, log=None):
        self.log = self.setup_log(log)
        self.log.info('Run initiated.')
        self.find_Optimum_bins = find_Optimum_bins
        self.bins = bins
        self.frame = None
        self.independent_vars = []
        self.independent_vars_type = []
        self.target_type = ''
        self.nrows = 0
        self.thresholds = {}
        self.bin_dict = {}


    def setup_log(self, log):
        if log is None:
            log = Logger()
        return log


    def basic_frame_check(self, frame, target, independent_vars):
        '''will put docstring later'''
        if independent_vars is None:
            self.log.info('All columns except target are considered as independent variables.')
            independent_vars = [column for column in frame.columns if column != target]
        if not isinstance(independent_vars, list):
            independent_vars = [independent_vars]
        self.independent_vars = copy.deepcopy(independent_vars)
        self.independent_vars_type = [self.column_type(frame, var, 'independent')
                                        for var in self.independent_vars]
        self.target_var = target
        self.target_type = self.column_type(frame, target, 'target')
        self.log.info('Target type: ' + self.target_type)
        all_cols = independent_vars
        all_cols.append(target)
        self.nrows = frame[target].count()
        self.frame = frame[all_cols].copy()


    def column_type(self, frame, var, var_type):
        '''put docstring later'''
        col_type = frame[var].dtype
        if var_type == 'independent':
            if col_type in ('int64', 'float64'):
                return 'continuous'
            else:
                return 'discrete'
        else:                                   # var_type == 'target'
            if col_type in ('int64', 'float64'):
                unique_sum = np.sum(frame[var].unique())
                if unique_sum in [0,1]:
                    return 'binary'
                else:
                    return 'continuous'
            else:
                raise TypeError('Target variable is not correct.')


    def run_DecisionTree(self, independent_var):
        if self.target_type == 'binary':
            self.log.info('Running DecisionTreeClassifier.')
            FineClassTree = DecisionTreeClassifier
        else:                                       # self.target_type == 'binary':
            self.log.info('Running DecisionTreeRegressor.')
            FineClassTree = DecisionTreeRegressor

        if self.find_Optimum_bins:
            self.log.info('Finding optimum bins.')
            optimum_bins = self.get_Optimum_bins()
        else:
            self.log.info('Using user given number of bins.')
            optimum_bins = self.bins - 1
        self.log.info('Number of bins: {0}'.format(optimum_bins + 1))

        min_samples_split = int(math.floor(self.nrows/(optimum_bins-1)))
        min_samples_leaf = divmod(min_samples_split, 3)[0]
        gini_tree = FineClassTree(
                                  #criterion = 'gini'for Classifier 'mse' for Regresor
                                  max_depth = None,
                                  min_samples_leaf = min_samples_leaf,
                                  min_samples_split = min_samples_split)
        self.log.info(gini_tree)
        d_tree = gini_tree.fit(self.frame[[independent_var]],
                               self.frame[[self.target_var]])
        thresholds = self.find_threshold(d_tree, node=0, thresholds=[])
        self.thresholds[independent_var] = self.sort_thresholds(thresholds)
        self.bin_dict[independent_var] = self.create_bin_dicts(self.thresholds[independent_var])
        self.log.info(self.bin_dict)


    def create_bin_dicts(self, thresholds):
        bin_dict = {'lower': np.append((-float('inf'),), thresholds),
                    'upper': np.append(thresholds, (float('inf'),)),
                   }
        bin_dict['labels'] = np.arange(1, len(bin_dict['lower'])+1)
        return bin_dict


    def sort_thresholds(self, thresholds):
        thresholds = list(set(thresholds))
        thresholds.sort()
        return thresholds


    def find_threshold(self, d_tree, node, thresholds):
        tree_ = d_tree.tree_
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            thresh = tree_.threshold[node]
            thresholds.append(thresh)
            self.find_threshold(d_tree, tree_.children_left[node], thresholds) #, depth + 1)
            thresholds.append(thresh)
            self.find_threshold(d_tree, tree_.children_right[node], thresholds) #, depth + 1)
        return thresholds


    def fit(self, frame, target, independent_vars=None):
        '''will put docstring later'''
        self.basic_frame_check(frame, target, independent_vars)
        for independent_var in self.independent_vars:
            self.run_DecisionTree(independent_var)
            part_create_label = partial(create_label,
                                        bin_dict = self.bin_dict[independent_var],
                                        value = independent_var)
            self.frame['labels'] = self.frame.apply(part_create_label, axis=1)        #SettingWithCopyWarning
            self.frame = _calc_stat(self.frame, target)


if __name__ == '__main__':
    data = pd.read_csv('../data/german_data.csv')
    fc = FineClass(find_Optimum_bins=False, bins=5)
    fc.fit(
            data,
            target='tar',
            independent_vars='credit_amount'
            # target='credit_amount',
            # independent_vars='age'
          )
    print fc.frame


