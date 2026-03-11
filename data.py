import os
import numpy as np
import torch
from torch.utils.data import Dataset

class ShapeNetPart(Dataset):
    def __init__(self, num_points=1024, partition='train'):
        self.num_points = num_points
        self.partition = partition
        # where the data is
        self.root = '/content/data_bin' 
        
        self.datapath = []
        

        if os.path.exists(self.root):
            categories = [d for d in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, d))]
            
            for cat in sorted(categories):
                pts_dir = os.path.join(self.root, cat, 'points')
                if os.path.exists(pts_dir):
                    # .npy files 
                    files = sorted([f for f in os.listdir(pts_dir) if f.endswith('.npy')])
                    for f in files:
                        self.datapath.append({
                            'point': os.path.join(pts_dir, f),
                            'label': os.path.join(self.root, cat, 'points_label', f),
                            'category': cat
                        })

        # Split Train/Test(80% / 20%)
        np.random.seed(42)
        indices = np.arange(len(self.datapath))
        np.random.shuffle(indices)
        
        split = int(len(self.datapath) * 0.8)
        if self.partition == 'train':
            self.active_indices = indices[:split]
        else:
            self.active_indices = indices[split:]

        print(f"Dataset {partition} of length {len(self.active_indices)} found")

    def __getitem__(self, index):

        fn = self.datapath[self.active_indices[index]]
        
        # 3. CHARGEMENT BINAIRE : Le CPU ne fait que copier les octets (Vitesse Max) a modifier
        pc = np.load(fn['point']).astype(np.float32)
        seg = np.load(fn['label']).astype(np.int64)
        
        # Subsampling
        # On choisit aléatoirement 'num_points' parmi les points disponibles ? unifrmement aussi à verifier
        choice = np.random.choice(len(seg), self.num_points, replace=True)
        pc = pc[choice, :]
        seg = seg[choice]
        
        # 5. Normalisation (Centrage et mise à l'échelle) 
        # pas sûre de cette étape
        pc = pc - np.mean(pc, axis=0)
        dist = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        if dist > 0:
            pc = pc / dist
            
        # Jittering
        if self.partition == 'train':
            noise = np.random.normal(0, 0.002, size=pc.shape)
            pc += noise

        # On transpose pour avoir [3, N] ce qui est attendu par le DGCNN à modifier
        return pc.transpose(1, 0).astype('float32'), 0, seg.astype('int64')

    def __len__(self):
        return len(self.active_indices)