import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR 
from model2 import DGCNN_PartSeg
from data3 import ShapeNetPart
from utils3 import calculate_shape_iou
from tqdm import tqdm 

def run_model(model, loader, criterion, device, cls_to_label, categories, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    
    total_loss = 0.0
    all_shape_ious = []
    per_category_ious = {i: [] for i in range(len(cls_to_label))}
    
    mode = "Train" if is_train else "Test "
    pbar = tqdm(enumerate(loader), total=len(loader), desc=mode, unit="batch", leave=False)
    
    for i, (data, label, seg) in pbar:
        data, label, seg = data.to(device), label.to(device), seg.to(device)
        batch_size = data.size(0)
        
        l_one_hot = torch.zeros(batch_size, 16).to(device)
        l_one_hot.scatter_(1, label.view(-1, 1), 1)

        if is_train: 
            optimizer.zero_grad()
        
        with torch.set_grad_enabled(is_train):
            logits = model(data, l_one_hot)
            loss = criterion(logits, seg)
            
            if is_train:
                loss.backward()
                optimizer.step()
        
        total_loss += loss.item()
        preds = logits.max(dim=1)[1]
        
        batch_ious = calculate_shape_iou(preds, seg, label, cls_to_label)
        all_shape_ious.extend(batch_ious)
        
        for idx, cat_id in enumerate(label.cpu().numpy()):
            per_category_ious[cat_id].append(batch_ious[idx])

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / len(loader)
    instance_miou = np.mean(all_shape_ious)
    
    # class mIou
    class_miou_list = []
    category_summary = {}
    for cat_id in range(len(cls_to_label)):
        if len(per_category_ious[cat_id]) > 0:
            mIoU_cat = np.mean(per_category_ious[cat_id])
            class_miou_list.append(mIoU_cat)
            category_summary[categories[cat_id]] = mIoU_cat
            
    class_miou = np.mean(class_miou_list)

    return avg_loss, instance_miou, class_miou, category_summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DGCNN Part Segmentation')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--num_points', type=int, default=2048)
    parser.add_argument('--k', type=int, default=20)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--model_path', type=str, default='checkpoints/best_model.pth')
    args = parser.parse_args()

    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = ShapeNetPart(args.num_points, 'train')
    test_dataset = ShapeNetPart(args.num_points, 'test')
    cls_to_label = train_dataset.get_seg_mapping()
    categories = train_dataset.categories

    model = DGCNN_PartSeg(k=args.k).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()

    if args.eval:
        if os.path.exists(args.model_path):
            model.load_state_dict(torch.load(args.model_path, map_location=device))
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
            loss, inst_m, cl_m, cat_dict = run_model(model, test_loader, criterion, device, cls_to_label, categories)
            print(f"\nFINAL EVAL | Instance mIoU: {inst_m:.4f} | Class mIoU: {cl_m:.4f}")
        else:
            print("Checkpoint introuvable.")
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.001) 
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

        best_miou = 0.0
        for epoch in range(args.epochs):
            # training
            tr_loss, tr_inst, tr_cl, _ = run_model(model, train_loader, criterion, device, cls_to_label, categories, optimizer)
            scheduler.step()
            
            # test
            te_loss, te_inst, te_cl, te_cats = run_model(model, test_loader, criterion, device, cls_to_label, categories)
            
            if te_inst > best_miou:
                best_miou = te_inst
                state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
                torch.save(state, 'checkpoints/best_model.pth')
            
            
            print(f"\n{'='*20} EPOCH {epoch+1:03d} {'='*20}")
            print(f"TRAIN | Loss: {tr_loss:.4f} | Inst mIoU: {tr_inst:.4f} | Class mIoU: {tr_cl:.4f}")
            print(f"TEST  | Loss: {te_loss:.4f} | Inst mIoU: {te_inst:.4f} | Class mIoU: {te_cl:.4f}")
            print(f"{'-'*50}")
            print(f"{'Test per category':<25} | {'mIoU':<10}")
            for cat_name, val in te_cats.items():
                print(f"{cat_name:<25} | {val:.4f}")
            print(f"{'='*50}\n")