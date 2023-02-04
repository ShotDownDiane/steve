import os
import torch 
import numpy as np

class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        if type(data) == torch.Tensor and type(self.mean) == np.ndarray:
            self.std = torch.from_numpy(self.std).to(data.device).type(data.dtype)
            self.mean = torch.from_numpy(self.mean).to(data.device).type(data.dtype)
        return (data * self.std) + self.mean

class MinMax01Scaler:

    """
    Standard the input
    """

    def __init__(self, min, max):
        self.min = min
        self.max = max

    def transform(self, data):
        return (data - self.min) / (self.max - self.min)

    def inverse_transform(self, data):
        if type(data) == torch.Tensor and type(self.min) == np.ndarray:
            self.min = torch.from_numpy(self.min).to(data.device).type(data.dtype)
            self.max = torch.from_numpy(self.max).to(data.device).type(data.dtype)
        return (data * (self.max - self.min) + self.min)

class MinMax11Scaler:
    """
    Standard the input
    """

    def __init__(self, min, max):
        self.min = min
        self.max = max

    def transform(self, data):
        return ((data - self.min) / (self.max - self.min)) * 2. - 1.

    def inverse_transform(self, data):
        if type(data) == torch.Tensor and type(self.min) == np.ndarray:
            self.min = torch.from_numpy(self.min).to(data.device).type(data.dtype)
            self.max = torch.from_numpy(self.max).to(data.device).type(data.dtype)
        return ((data + 1.) / 2.) * (self.max - self.min) + self.min

# def STDataloader(X, Y, batch_size, shuffle=True, drop_last=True):
#     cuda = True if torch.cuda.is_available() else False
#     TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
#     X, Y = TensorFloat(X), TensorFloat(Y)
#     data = torch.utils.data.TensorDataset(X, Y)
#     dataloader = torch.utils.data.DataLoader(
#         data, 
#         batch_size=batch_size,
#         shuffle=shuffle, 
#         drop_last=drop_last,
#     )
#     return dataloader

def STDataloader_T(X, Y, time_label, c, batch_size, device,shuffle=True, drop_last=True,train_flag=True):

    TensorFloat = torch.FloatTensor
    TensorInt = torch.LongTensor
    # time_label = np.load('/home/zhangwt/git-share/workspace/HeST/data/NYCBike1/time_label.npz')['y_label']
    if train_flag:
        X, Y ,time_label,c= TensorFloat(X).to(device), TensorFloat(Y).to(device), TensorInt(time_label).to(device), TensorFloat(c).to(device)
        data = torch.utils.data.TensorDataset(X, Y, time_label, c)
    else:
        X, Y ,c = TensorFloat(X).to(device), TensorFloat(Y).to(device), TensorFloat(c).to(device)
        data = torch.utils.data.TensorDataset(X, Y, c)

    dataloader = torch.utils.data.DataLoader(
        data,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )
    return dataloader

def normalize_data(data, scalar_type='Standard'):
    scalar = None
    if scalar_type == 'MinMax01':
        scalar = MinMax01Scaler(min=data.min(), max=data.max())
    elif scalar_type == 'MinMax11':
        scalar = MinMax11Scaler(min=data.min(), max=data.max())
    elif scalar_type == 'Standard':
        scalar = StandardScaler(mean=data.mean(), std=data.std())
    else:
        raise ValueError('scalar_type is not supported in data_normalization.')
    # print('{} scalar is used!!!'.format(scalar_type))
    # time.sleep(3)
    return scalar

def get_dataloader(data_dir, dataset, batch_size, test_batch_size, device, scalar_type='Standard'):
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(data_dir, dataset, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']
        data['time_'+category] = cat_data['time_label'] 
        data['c_'+category] = cat_data['c']
        # print(category, data['x_' + category].shape)
    scaler = normalize_data(np.concatenate([data['x_train'], data['x_val']], axis=0), scalar_type)
    # scaler = StandardScaler(mean=data['x_train'].mean(), std=data['x_train'].std())
    # Data format
    for category in ['train', 'val', 'test']:
        data['x_' + category] = scaler.transform(data['x_' + category])
        data['y_' + category] = scaler.transform(data['y_' + category]) # y 其实不需要transform
    # Construct dataloader
    dataloader = {}
    dataloader['train'] = STDataloader_T(
        data['x_train'],
        data['y_train'],
        data['time_train'],
        data['c_train'],
        batch_size,
        device=device,
        shuffle=True
    )

    dataloader['val'] = STDataloader_T(
        data['x_val'], 
        data['y_val'],
        data['time_val'],
        data['c_val'],
        test_batch_size,
        device=device, 
        shuffle=False
    )
    dataloader['test'] = STDataloader_T(
        data['x_test'], 
        data['y_test'],
        None,
        data['c_test'], 
        test_batch_size,
        device=device, 
        shuffle=False, 
        drop_last=False,
        train_flag=False
    )
    dataloader['scaler'] = scaler
    return dataloader

if __name__ == '__main__':
    loader = get_dataloader('../data/', 'BikeNYC', batch_size=64, test_batch_size=64)
    for key in loader.keys():
        print(key)