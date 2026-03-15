import os
import numpy as np
import torch
from torch.utils.data import Dataset

def translate_pointcloud(pointcloud):
    # random scaling 
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3]) 
    # random translation 
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
    
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud

class ShapeNetPart(Dataset):
    def __init__(self, num_points=1024, partition='train'):
        self.num_points = num_points
        self.partition = partition
        self.root = '/content/data_bin' 
        
        self.datapath = []
        self.cat_to_id = {}
        self.categories = []

        if os.path.exists(self.root):
            self.categories = sorted([d for d in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, d))])
            
            for i, cat in enumerate(self.categories):
                self.cat_to_id[cat] = i 
                pts_dir = os.path.join(self.root, cat, 'points')
                if os.path.exists(pts_dir):
                    files = sorted([f for f in os.listdir(pts_dir) if f.endswith('.npy')])
                    for f in files:
                        self.datapath.append({
                            'point': os.path.join(pts_dir, f),
                            'label': os.path.join(self.root, cat, 'points_label', f),
                            'category_id': i 
                        })

        # Split Train/Test (80% / 20%) 
        np.random.seed(42)
        indices = np.arange(len(self.datapath))
        np.random.shuffle(indices)
        
        split = int(len(self.datapath) * 0.8)
        if self.partition == 'train':
            self.active_indices = indices[:split]
        else:
            self.active_indices = indices[split:]

        print(f"Dataset {partition} of length {len(self.active_indices)} found. ({len(self.categories)} catégories)")

    def get_seg_mapping(self):
        """
        to get IDs for categs
        """
        mapping = [set() for _ in range(len(self.categories))]
        
        

        for fn in self.datapath:
            cat_id = fn['category_id']
            seg = np.load(fn['label'])
            mapping[cat_id].update(np.unique(seg))
            

        return [sorted(list(s)) for s in mapping]

    def __getitem__(self, index):
        fn = self.datapath[self.active_indices[index]]
        
    
        pc = np.load(fn['point']).astype(np.float32)
        seg = np.load(fn['label']).astype(np.int64)
        cat_id = fn['category_id'] 
        
        # Subsampling 
        if len(seg) >= self.num_points:
            choice = np.random.choice(len(seg), self.num_points, replace=False)
        else:
            choice = np.random.choice(len(seg), self.num_points, replace=True)
             
        pc = pc[choice, :]
        seg = seg[choice]
        
        # Normalisation 
        pc = pc - np.mean(pc, axis=0)
        dist = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        if dist > 0:
            pc = pc / dist
            
        # Augmentation (only for training)
        if self.partition == 'train':
             pc = translate_pointcloud(pc)

        return pc.transpose(1, 0).astype('float32'), cat_id, seg.astype('int64')

    def __len__(self):
        return len(self.active_indices)