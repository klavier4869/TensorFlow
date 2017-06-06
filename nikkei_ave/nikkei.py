""" predicted closing price after 1day using data 30days ago.
    csv data is descending by date
    - train/test data format
     | month | day | opening | hight | low | closing |
    - Slide 31 pieces of data.
     train_data: 2010-2016 (1712 days worth) = num of 1682
     test_data: 2017 (104 days worth) = num of 74
"""
import os.path
import os
import numpy as np


class NikkeiData:
    def __init__(self):
        self.dataset = {}
        self.test = {} # dict for cache
        self.in_days = 30
        self.elem_size = 6 # num of one day size
        self.predict_index = 5 # closing index
        self.in_size = self.elem_size * self.in_days
        self.out_size = 1

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
        """ fetch train data randomly using batch_mask. """
        train = {}
        train_size = len(self.dataset['train']) - self.in_days # in_days = slide size
        batch_mask = np.random.choice(train_size, batch_size)
        train['input'] = np.array(
                        [self.dataset['train'][v : v + self.in_days] for v in batch_mask])
        train['output'] = np.array(
                        [self.dataset['train'][v + self.in_days][self.predict_index] for v in batch_mask])
        train['output'] = train['output'].reshape(-1, self.out_size) # cos its shape of used output in tensorflow
        if isFlatten:
            train['input'] = train['input'].reshape(-1, self.in_size)
        return train['input'], train['output']

    def fetch_test(self, isFlatten=True):
        """ test data fetch all data. """
        # cache the test data
        if len(self.test) == 0:
            test_size = len(self.dataset['test']) - self.in_days
            self.test['input'] = np.array(
                                [self.dataset['test'][v : v + self.in_days] for v in range(test_size)])
            self.test['output'] = np.array(
                                [self.dataset['test'][v + self.in_days][self.predict_index] for v in range(test_size)])
            self.test['output'] = self.test['output'].reshape(-1, self.out_size)
        if isFlatten:
            self.test['input'] = self.test['input'].reshape(-1, self.in_size)
        return self.test['input'], self.test['output']
