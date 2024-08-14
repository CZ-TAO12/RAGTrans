import random
import numpy as np
import torch
from torch.autograd import Variable
import pickle
from torch_geometric.data import Dataset
from torch_geometric.data import Data

class Options(object):
    def __init__(self, data_name='WEIXIN2'):
        self.data = 'data/' + data_name + '/all_data.txt'
        self.category = 'data/' + data_name + '/train_category.json'

        self.net_data = 'data/' + data_name + '/edges.txt'

        # train file path.
        self.train_data = 'data/' + data_name + '/train_data.pickle'
        # valid file path.
        self.valid_data = 'data/' + data_name + '/val_data.pickle'
        # test file path.
        self.test_data = 'data/' + data_name + '/test_data.pickle'

def _readFromFile(filename):
    graph_data = []
    all_data = pickle.load(open(filename, 'rb'))
    ####process the raw data
    for line in all_data:
        pid = torch.LongTensor([int(line[0])])
        uid = torch.LongTensor([int(line[-1])])
        popularity = torch.FloatTensor([float(line[1])])
        related_items = torch.LongTensor(np.array(line[2]))
        hg_idx = torch.LongTensor(np.array(line[3]))
        hg_data = Data(edge_index=hg_idx, y=popularity, related_items=related_items, pid=pid, uid=uid)
        graph_data.append(hg_data)

    return graph_data

def Read_data(data_name, with_EOS=True):

    options = Options(data_name)

    train = _readFromFile(options.train_data)
    valid = _readFromFile(options.valid_data)
    test = _readFromFile(options.test_data)

    return train, valid, test



class datasets(Dataset):
    def __init__(self, data):

        self.item_num = len(data)
        self.data = data

    def __len__(self):
        return self.item_num

    def __getitem__(self, index):
        # pid = self.data[index][0]
        # uid = self.data[index][1]
        # popularity = self.data[index][-1]
        # img_cluser = self.data[index][3]
        # related_items = np.array([int(x) for x in self.data[index][-1]])

        return self.data[index]

    def len(self):
        return self.item_num

    def get(self, index):
        return self.data[index]


