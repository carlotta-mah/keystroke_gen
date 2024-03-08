from torch.utils.data import Dataset
class KeystrokeDataset(Dataset):
    '''Custom dataset that is used to save processed keystroke data'''
    def __init__(self, data, labels):
        if data is None:
            data = []
        self.data = data
        if labels is None:
            labels = []
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        labels = self.labels[idx]
        return data, labels

    def append_item(self, data_point, label):
        '''append a data_point and its label to the dataset'''
        self.data.append(data_point)
        self.labels.append(label)
