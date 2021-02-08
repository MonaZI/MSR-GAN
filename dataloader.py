import torch
import torch.utils.data as data

class Dataset(data.Dataset):
    def __init__(self, meas, args, shifts=None):
        """
        The class for the dataset.
        :param meas: The set of measurements
        :param args: a set of arguments
        """
        self.args = args
        self.meas = meas
        self.num_samples = len(self.meas)
        if not (shifts is None):
            self.shifts = shifts
        print('The number of samples: %d' %(self.num_samples))

    def __getitem__(self, index):
        return self.meas[index, :]#, self.shifts[index]

    def __len__(self):
        return self.num_samples


def collate_fn(data):
    """
    Converts the batched data in torch tensor and returns them
    :param data: the data in form of numpy array
    :return: the data in torch.tensor
    """
    #meas, shifts = zip(*data)
    meas = data
    # adding the dimensionality corresponding to the number of channels
    # here for the input, we only have one channel
    meas = torch.tensor(meas).unsqueeze(1).float()
    #shifts = torch.tensor(shifts).unsqueeze(1).float()
    return meas#, shifts


def get_loader(dataset, args, is_test=False):
    """
    Creates the dataloader from dataset based on the given arguments
    :param dataset: the dataset
    :param args: the required arguments
    :param is_test: whether it is the train or test data
    :return: the dataloader
    """
    dataLoader = data.DataLoader(dataset=dataset, batch_size=args.batch_size, num_workers=0, collate_fn=collate_fn,
                                 shuffle=True if (is_test==False) else False)
    return dataLoader
