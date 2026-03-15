import torch
import numpy as np

def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_dims = x.size(1)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    
    if idx is None:
        inner = -2 * torch.matmul(x.transpose(2, 1), x)
        xx = torch.sum(x**2, dim=1, keepdim=True)
        dist = -xx - inner - xx.transpose(2, 1)
        idx = dist.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)

    device = x.device 
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
 
    feature = x.transpose(2, 1).contiguous()
    neighbor_features = feature.view(batch_size * num_points, -1)[idx, :]
    neighbor_features = neighbor_features.view(batch_size, num_points, k, num_dims)
    central_features = feature.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((central_features, neighbor_features - central_features), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature

def calculate_shape_iou(preds, targets, category_ids, cls_to_label):

    preds = preds.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    category_ids = category_ids.detach().cpu().numpy()
    
    shape_ious = []

    for i in range(preds.shape[0]):
        cat = category_ids[i]
        valid_labels = cls_to_label[cat]

        part_ious = []

        for part in valid_labels:

            I = np.sum((preds[i] == part) & (targets[i] == part))
            U = np.sum((preds[i] == part) | (targets[i] == part))

            if U == 0:
                iou = 1.0
            else:
                iou = I / float(U)

            part_ious.append(iou)

        shape_ious.append(np.mean(part_ious))

    return shape_ious