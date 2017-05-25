""" predicted closing price after 1day using data 30days ago
    data format
    | year | month | day | opening | hight | low | closing |
    train_data: 2015
    test_data: 2016
"""
import os.path
import os
import csv
import numpy as np


class NikkeiData:
    def __init__(self):
        self.dataset = {}
        self.test = {}
        self.in_size = 30 # 30days
        self.out_size = 1 # 1day
        self._load_data()

    def _load_data(self):
        dataset_dir = os.path.dirname(os.path.abspath(__file__))
        file_names = {'train': dataset_dir + '/train_data.csv',
                        'test': dataset_dir + '/test_data.csv'}
        def format_data(array):
            """ formatting date data and convert type to float """
            data = array.pop(0)
            data = data.split('-')
            data.extend(array)
            return np.array(data, np.float64)
        read_data = {}
        for k in file_names:
            with open(file_names[k], 'rt') as f:
                csv_obj = csv.reader(f)
                read_data[k] = np.array([format_data(v) for v in csv_obj])
        self.dataset = read_data

    def fetch_train(self, batch_size=100):
        train = {}
        offset = self.in_size + self.out_size # train size offset
        train_size = len(self.dataset['train']) - offset
        batch_mask = np.random.choice(train_size, batch_size)
        train['input'] = np.array([self.dataset['train'][v : v+30] for v in batch_mask])
        train['output'] = np.array([self.dataset['train'][v+31][6] for v in batch_mask])
        return train
    def fetch_test(self):
        """ cahce test data """
        if len(self.test) > 0:
            return self.test
        offset = self.in_size + self.out_size # test size offset
        test_size = len(self.dataset['test']) - offset
        self.test['input'] = np.array([self.dataset['test'][v : v+30] for v in range(test_size)])
        self.test['output'] = np.array([self.dataset['test'][v][6] for v in range(test_size)])
        return self.test
