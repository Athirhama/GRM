import os
import numpy as np
import torch
from torch.utils.data import Dataset
import glob

class ShapeNetPart(Dataset):
    def __init__(self, num_points=1024, partition='train'):
        self.num_points = num_points
        self.partition = partition
        # On revient à la racine qui contient toutes les catégories (02691156, 02773838, etc.)
        self.root = '/content/GRM/data/shapenet_part_seg_hdf5_data/PartAnnotation'
        
        self.datapath = []
        
        # Liste de toutes les catégories présentes dans le dossier
        categories = [d for d in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, d))]
        print(f"📂 Catégories détectées : {len(categories)}")

        for cat in categories:
            cat_dir = os.path.join(self.root, cat)
            pts_dir = os.path.join(cat_dir, 'points')
            lbl_dir = os.path.join(cat_dir, 'points_label')
            
            if os.path.exists(pts_dir):
                pts_files = [f for f in os.listdir(pts_dir) if f.endswith('.pts')]
                for f in pts_files:
                    p_file = os.path.join(pts_dir, f)
                    base_name = f.replace('.pts', '')
                    
                    # Recherche récursive pour gérer les sous-dossiers (body, wing, etc.)
                    label_search = glob.glob(os.path.join(lbl_dir, '**', base_name + '.seg'), recursive=True)
                    
                    if label_search:
                        self.datapath.append({'point': p_file, 'label': label_search[0]})

        # Split Train/Test global (80/20)
        np.random.seed(42)
        indices = np.arange(len(self.datapath))
        np.random.shuffle(indices)
        split = int(len(self.datapath) * 0.8)
        
        self.active_indices = indices[:split] if partition == 'train' else indices[split:]
        print(f"✅ Dataset complet chargé : {len(self.active_indices)} fichiers pour la phase {partition}.")

    def __getitem__(self, index):
        fn = self.datapath[self.active_indices[index]]
        pc = np.genfromtxt(fn['point']).astype(np.float32)
        seg = np.genfromtxt(fn['label']).astype(np.int64)
        
        # Subsampling
        choice = np.random.choice(len(seg), self.num_points, replace=True)
        pc, seg = pc[choice, :], seg[choice]
        
        # Normalisation
        pc = pc - np.mean(pc, axis=0)
        dist = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        if dist > 0: pc = pc / dist
        
        # Data Augmentation (Jittering)
        if self.partition == 'train':
            pc += np.random.normal(0, 0.01, size=pc.shape)

        return pc.transpose(1, 0).astype('float32'), 0, seg.astype('int64')

    def __len__(self):
        return len(self.active_indices)