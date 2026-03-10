import argparse, os, torch, time
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR # <--- Pour le Cosine Annealing
from model import DGCNN_PartSeg
from data import ShapeNetPart
from utils import calculate_shape_iou
from tqdm import tqdm 

def run_model(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss, total_iou = 0, 0
    
    mode = "🚀 Training" if is_train else "🔍 Eval"
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
            loss = criterion(logits.view(-1, 50), seg.view(-1))
            
            if is_train:
                loss.backward()
                optimizer.step()
        
        total_loss += loss.item()
        preds = logits.max(dim=1)[1]
        total_iou += calculate_shape_iou(preds, seg)
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / len(loader), total_iou / len(loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DGCNN Part Segmentation')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size (Paper: 32)')
    parser.add_argument('--epochs', type=int, default=200, help='Nombre d\'époques (Paper: 200)')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate (Paper: 0.1)')
    parser.add_argument('--num_points', type=int, default=2048, help='Points')
    parser.add_argument('--k', type=int, default=20, help='Voisins')
    # ... autres arguments ...
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DGCNN_PartSeg(k=args.k).to(device)
    
    # --- CONFIGURATION PAPIER ORIGINAL ---
    # 1. SGD avec momentum 0.9
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    
    # 2. Cosine Annealing (descend de 0.1 à 0.001)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.001)
    
    criterion = nn.CrossEntropyLoss()

    # --- DATALOADER ---
    train_loader = DataLoader(
        ShapeNetPart(args.num_points, 'train'), 
        batch_size=args.batch_size, 
        shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True
    )

    best_iou = 0
    for epoch in range(args.epochs):
        # Entraînement
        loss, iou = run_model(model, train_loader, criterion, device, optimizer)
        
        # Mise à jour du Learning Rate (Cosine Annealing)
        scheduler.step() 
        
        current_lr = optimizer.param_groups[0]['lr']
        is_best = iou > best_iou
        if is_best:
            best_iou = iou
            torch.save(model.state_dict(), 'checkpoints/best_model.pth')
            
        print(f"📅 Epoch [{epoch+1:03d}/{args.epochs}] | Loss: {loss:.4f} | mIoU: {iou:.4f} | LR: {current_lr:.6f} | {'⭐' if is_best else ''}")