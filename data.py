import os, glob, h5py
import numpy as np
from torch.utils.data import Dataset

def load_data_partseg(partition):
    all_data, all_label, all_seg = [], [], []
    path = os.path.join('data', 'shapenet_part_seg_hdf5_data', f'{partition}*.h5')
    files = glob.glob(path)
    for h5_name in files:
        with h5py.File(h5_name, 'r') as f:
            all_data.append(f['data'][:])
            all_label.append(f['label'][:])
            all_seg.append(f['pid'][:])
    return np.concatenate(all_data), np.concatenate(all_label), np.concatenate(all_seg)

class ShapeNetPart(Dataset):
    def __init__(self, num_points, partition='train'):
        self.data, self.label, self.seg = load_data_partseg(partition)
        self.num_points = num_points
        self.partition = partition

    def __getitem__(self, item):
        pc, label, seg = self.data[item], self.label[item], self.seg[item]
        # Uniform Sampling
        idx = np.random.choice(len(pc), self.num_points, replace=True)
        pc, seg = pc[idx], seg[idx]

        if self.partition == 'train':
            pc *= np.random.uniform(0.8, 1.2) # Scaling
            pc += np.random.normal(0, 0.01, size=pc.shape) # Jitter

        return pc.transpose(1, 0).astype('float32'), label, seg.astype('int64')

    def __len__(self):
        return len(self.data)