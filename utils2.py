import torch
import numpy as np

# --- Dictionnaires officiels pour ShapeNet Part (16 catégories, 50 parties) ---
# seg_num : Nombre de parties pour chaque catégorie (ex: Avion (0) = 4 parties)
seg_num = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
# index_start : L'index de départ des parties pour chaque catégorie (ex: Avion commence à 0, Sac commence à 4)
index_start = [0, 4, 6, 8, 12, 16, 19, 22, 24, 28, 30, 36, 38, 41, 44, 47]

def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_dims = x.size(1)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    
    if idx is None:
        # Compute pairwise distance
        inner = -2 * torch.matmul(x.transpose(2, 1), x)
        xx = torch.sum(x**2, dim=1, keepdim=True)
        dist = -xx - inner - xx.transpose(2, 1)
        idx = dist.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
 
    feature = x.transpose(2, 1).contiguous()
    neighbor_features = feature.view(batch_size * num_points, -1)[idx, :]
    neighbor_features = neighbor_features.view(batch_size, num_points, k, num_dims)
    central_features = feature.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    # Concatenate [xi, xj - xi]
    feature = torch.cat((central_features, neighbor_features - central_features), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature

def calculate_shape_IoU(pred_np, seg_np, label, class_choice=None, visual=False):
    # Sécurité : on détache et on convertit en Numpy si ce sont des tenseurs PyTorch
    if torch.is_tensor(pred_np): pred_np = pred_np.cpu().detach().numpy()
    if torch.is_tensor(seg_np): seg_np = seg_np.cpu().detach().numpy()
    if torch.is_tensor(label): label = label.cpu().detach().numpy()

    if not visual:
        label = label.squeeze()
        
    shape_ious = []
    # On boucle sur chaque objet du batch
    for shape_idx in range(seg_np.shape[0]):
        if not class_choice:
            # On récupère les identifiants de parties valides pour LA catégorie de cet objet
            start_index = index_start[label[shape_idx]]
            num = seg_num[label[shape_idx]]
            parts = range(start_index, start_index + num)
        else:
            parts = range(seg_num[label[0]])
            
        part_ious = []
        # On calcule l'IoU uniquement sur ces parties valides
        for part in parts:
            I = np.sum(np.logical_and(pred_np[shape_idx] == part, seg_np[shape_idx] == part))
            U = np.sum(np.logical_or(pred_np[shape_idx] == part, seg_np[shape_idx] == part))
            if U == 0:
                iou = 1  # L'objet n'a pas cette partie, et le réseau n'a rien prédit : 100% de réussite !
            else:
                iou = I / float(U)
            part_ious.append(iou)
            
        # Moyenne des parties pour cet objet (Instance IoU)
        shape_ious.append(np.mean(part_ious))
        
    # Retourne une liste contenant l'IoU de chaque objet du batch
    return shape_ious
