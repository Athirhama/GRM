import os
import numpy as np
import torch
from torch.utils.data import Dataset

class ShapeNetPart(Dataset):
    def __init__(self, num_points=1024, partition='train'):
        self.num_points = num_points
        self.partition = partition
        # On pointe vers le dossier où tu as fait la conversion
        self.root = '/content/data_bin' 
        
        self.datapath = []
        
        # 1. On scanne les dossiers convertis
        if os.path.exists(self.root):
            categories = [d for d in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, d))]
            
            for cat in sorted(categories):
                pts_dir = os.path.join(self.root, cat, 'points')
                if os.path.exists(pts_dir):
                    # On liste les fichiers .npy (beaucoup plus rapide que .pts)
                    files = sorted([f for f in os.listdir(pts_dir) if f.endswith('.npy')])
                    for f in files:
                        self.datapath.append({
                            'point': os.path.join(pts_dir, f),
                            'label': os.path.join(self.root, cat, 'points_label', f),
                            'category': cat
                        })

        # 2. Split Train/Test déterministe (80% / 20%)
        np.random.seed(42)
        indices = np.arange(len(self.datapath))
        np.random.shuffle(indices)
        
        split = int(len(self.datapath) * 0.8)
        if self.partition == 'train':
            self.active_indices = indices[:split]
        else:
            self.active_indices = indices[split:]

        print(f"✅ Dataset {partition} : {len(self.active_indices)} objets chargés en mode binaire.")

    def __getitem__(self, index):
        # Récupération du chemin via l'index du split
        fn = self.datapath[self.active_indices[index]]
        
        # 3. CHARGEMENT BINAIRE : Le CPU ne fait que copier les octets (Vitesse Max)
        pc = np.load(fn['point']).astype(np.float32)
        seg = np.load(fn['label']).astype(np.int64)
        
        # 4. Échantillonnage (Subsampling)
        # On choisit aléatoirement 'num_points' parmi les points disponibles
        choice = np.random.choice(len(seg), self.num_points, replace=True)
        pc = pc[choice, :]
        seg = seg[choice]
        
        # 5. Normalisation (Centrage et mise à l'échelle)
        pc = pc - np.mean(pc, axis=0)
        dist = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        if dist > 0:
            pc = pc / dist
            
        # 6. Data Augmentation légère pour l'entraînement
        if self.partition == 'train':
            # Petit bruit gaussien (Jittering) pour la robustesse
            noise = np.random.normal(0, 0.002, size=pc.shape)
            pc += noise

        # Retourne (Points, Catégorie_ID, Segmentation_Labels)
        # On transpose pour avoir [3, N] ce qui est attendu par le DGCNN
        return pc.transpose(1, 0).astype('float32'), 0, seg.astype('int64')

    def __len__(self):
        return len(self.active_indices)