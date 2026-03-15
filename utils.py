import torch
import numpy as np

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

def calculate_shape_iou(preds, targets, num_parts=50):
    """ Calculates the Intersection over Union for point cloud segments """
    iou_list = []
    preds = preds.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    
    for i in range(num_parts):
        if np.sum(targets == i) > 0: # Only if the part exists in the ground truth
            intersection = np.sum((preds == i) & (targets == i))
            union = np.sum((preds == i) | (targets == i))
            iou_list.append(intersection / union)
    return np.mean(iou_list) if iou_list else 0
