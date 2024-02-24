from torch.utils.data import Dataset
class KeystrokeDataset(Dataset):
    def __init__(self, data, labels):
        if data is None:
            data = []
        self.data = data
        self.labels = labels
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        labels = self.labels.iloc[idx]
        return data, labels

    def append_item(self, data_point, label):
        self.data.append(data_point)
        self.labels.append(label)
