import torch
import numpy as np

def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_dims = x.size(1)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    
    if idx is None:
        # Calcul de la distance euclidienne par paire : ||a-b||^2 = ||a||^2 + ||b||^2 - 2ab
        inner = -2 * torch.matmul(x.transpose(2, 1), x)
        xx = torch.sum(x**2, dim=1, keepdim=True)
        dist = -xx - inner - xx.transpose(2, 1)
        idx = dist.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)

    device = x.device # Plus robuste que de redéfinir device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
 
    feature = x.transpose(2, 1).contiguous()
    neighbor_features = feature.view(batch_size * num_points, -1)[idx, :]
    neighbor_features = neighbor_features.view(batch_size, num_points, k, num_dims)
    central_features = feature.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    # Concatenate [xi, xj - xi] : l'essence de EdgeConv
    feature = torch.cat((central_features, neighbor_features - central_features), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature

def calculate_shape_iou(preds, targets, category_ids, cls_to_label):
    """
    Calcul rigoureux de l'Instance mIoU.
    preds: [B, N] (predictions du modèle)
    targets: [B, N] (vérité terrain)
    category_ids: [B] (ID de la catégorie de l'objet, ex: 0 pour Airplane)
    cls_to_label: Liste de listes contenant les IDs de segments valides par catégorie.
    """
    preds = preds.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    category_ids = category_ids.detach().cpu().numpy()
    
    batch_ious = []
    
    # On itère sur chaque objet du batch individuellement
    for i in range(preds.shape[0]):
        cat = category_ids[i]
        valid_labels = cls_to_label[cat] # On récupère les segments possibles pour cette catégorie
        
        parts_iou = []
        for part in valid_labels:
            # On ne calcule l'IoU que si la partie est présente dans la cible (standard ShapeNet)
            if np.sum(targets[i] == part) > 0:
                intersection = np.sum((preds[i] == part) & (targets[i] == part))
                union = np.sum((preds[i] == part) | (targets[i] == part))
                parts_iou.append(intersection / union)
        
        # Moyenne des parties pour cet objet précis
        if parts_iou:
            batch_ious.append(np.mean(parts_iou))
        else:
            batch_ious.append(0.0)
            
    return batch_ious # On renvoie une liste d'IoU (un par objet)