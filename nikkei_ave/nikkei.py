""" predicted closing price after 1day using data 30days ago
    csv data is descending by date
    - train/test data format
     | date offset | opening | hight | low | closing |
     date offset: 30 1day ago, 29 2days ago ...
    - Slide 31 pieces of data.
     train_data: 1984-2001 (3698 days worth) = num: 3668
     test_data: 2002-2017 (3779 days worth) = num: 3749
"""
import os.path
import os
import numpy as np


class NikkeiData:
    def __init__(self):
        self.dataset = {}
        self.test = {} # for cache
        self.in_size = 30 # 30days
        self.out_size = 1 # 1day
        self.date_offset =  np.arange(30, 0, -1)

        self._load_data()

    def _load_data(self):
        dataset_dir = os.path.dirname(os.path.abspath(__file__))
        file_names = {'train': dataset_dir + '/train_data.csv',
                        'test': dataset_dir + '/test_data.csv'}
        read_data = {}
        for k in file_names:
            read_data[k] = np.loadtxt(file_names[k], delimiter=',', dtype='float32')
        self.dataset = read_data

    def fetch_train(self, batch_size=50, isFlatten=True):
        """ fetch train data randomly using batch_mask
            data processing
             - add date infomation to input data
             - reshape output data cos needs to be (-1, 1)
        """
        train = {}
        train_size = len(self.dataset['train']) - 30 # 30 is slide size
        batch_mask = np.random.choice(train_size, batch_size)
        train['input'] = [self.dataset['train'][v : v+30] for v in batch_mask]
        train['input'] = np.array([np.c_[self.date_offset, v] for v in train['input']])
        train['output'] = np.array([self.dataset['train'][v+30][3] for v in batch_mask])
        train['output'] = train['output'].reshape(-1, 1)
        if isFlatten:
            train['input'] = train['input'].reshape(-1, 150) # 150 is one batch size
        return train['input'], train['output']

    def fetch_test(self, isFlatten=True):
        """ test data fetch all data
            data processing contents same as fetch_train
        """
        # cache test data
        if len(self.test) == 0:
            test_size = len(self.dataset['test']) - 30
            self.test['input'] = [self.dataset['test'][v : v+30] for v in range(test_size)]
            self.test['input'] = np.array([np.c_[self.date_offset, v] for v in self.test['input']])
            self.test['output'] = np.array([self.dataset['test'][v+30][3] for v in range(test_size)])
            self.test['output'] = self.test['output'].reshape(-1, 1)
        if isFlatten:
            self.test['input'] = self.test['input'].reshape(-1, 150)
        return self.test['input'], self.test['output']
