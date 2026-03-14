import os
import numpy as np
import torch
from torch.utils.data import Dataset

def translate_pointcloud(pointcloud):
    # scaling aléatoire
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3]) 
    # translation aléatoire
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
  
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud

class ShapeNetPart(Dataset):
    def __init__(self, num_points=1024, partition='train'):
        self.num_points = num_points
        self.partition = partition
        # where the data is
        self.root = '/content/data_bin' 
        
        self.datapath = []
        # dictionnaire pour transformer les chaines de caractères de catégories en ID
        self.cat_to_id = {}

        if os.path.exists(self.root):
            categories = sorted([d for d in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, d))])
            
            for i, cat in enumerate(categories):
                # assignation d'un entier à chaque catégorie
                self.cat_to_id[cat] = i 
                pts_dir = os.path.join(self.root, cat, 'points')
                if os.path.exists(pts_dir):
                    # .npy files 
                    files = sorted([f for f in os.listdir(pts_dir) if f.endswith('.npy')])
                    for f in files:
                        self.datapath.append({
                            'point': os.path.join(pts_dir, f),
                            'label': os.path.join(self.root, cat, 'points_label', f),
                            # on garde l'ID plutôt que la chaine de caractère et comme ça ça sera utilisable dans le get_item
                            'category_id': i 
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

        print(f"Dataset {partition} of length {len(self.active_indices)} found. ({len(categories)} catégories)")

    def __getitem__(self, index):

        fn = self.datapath[self.active_indices[index]]
        
        # 3. CHARGEMENT BINAIRE
        pc = np.load(fn['point']).astype(np.float32)
        seg = np.load(fn['label']).astype(np.int64)
        # NOUVEAU : Récupération de la vraie catégorie de l'objet
        cat_id = fn['category_id'] 
        
        # Subsampling
        # On choisit aléatoirement 'num_points' parmi les points disponibles ? unifrmement aussi à verifier (j'ai passé le replace en False)
        if len(seg) >= self.num_points:
            choice = np.random.choice(len(seg), self.num_points, replace=False)
        else:
             # Fallback au cas où un très petit objet aurait moins de points que self.num_points
             choice = np.random.choice(len(seg), self.num_points, replace=True)
             
        pc = pc[choice, :]
        seg = seg[choice]
        
        # 5. Normalisation (Centrage et mise à l'échelle dans une sphère unité) 
        pc = pc - np.mean(pc, axis=0)
        dist = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        if dist > 0:
            pc = pc / dist
            
        # Data Augmentation de translation
        if self.partition == 'train':
             pc = translate_pointcloud(pc)

        # On transpose pour avoir [3, N] ce qui est attendu par le DGCNN
        # j'ai rajouté cette fois-ci l'ID
        return pc.transpose(1, 0).astype('float32'), cat_id, seg.astype('int64')

    def __len__(self):
        return len(self.active_indices)
