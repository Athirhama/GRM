import argparse, os, torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import DGCNN_PartSeg
from data import ShapeNetPart
from utils import calculate_shape_iou

def run_model(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    
    total_loss, total_iou = 0, 0
    
    for data, label, seg in loader:
        data, label, seg = data.to(device), label.to(device), seg.to(device)
        batch_size = data.size(0)
        
        # One-hot categorical vector
        l_one_hot = torch.zeros(batch_size, 16).to(device)
        l_one_hot.scatter_(1, label.view(-1, 1), 1)

        if is_train: optimizer.zero_grad()
        
        with torch.set_grad_enabled(is_train):
            logits = model(data, l_one_hot)
            loss = criterion(logits.view(-1, 50), seg.view(-1))
            if is_train:
                loss.backward()
                optimizer.step()
        
        total_loss += loss.item()
        preds = logits.max(dim=1)[1]
        total_iou += calculate_shape_iou(preds, seg)

    return total_loss / len(loader), total_iou / len(loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--model_path', type=str, default='checkpoints/best_model.pth')
    parser.add_argument('--num_points', type=int, default=1024)
    parser.add_argument('--k', type=int, default=20)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DGCNN_PartSeg(k=args.k).to(device)
    criterion = nn.CrossEntropyLoss()

    if args.eval:
        model.load_state_dict(torch.load(args.model_path))
        loader = DataLoader(ShapeNetPart(args.num_points, 'test'), batch_size=32)
        loss, iou = run_model(model, loader, criterion, device)
        print(f"Evaluation Complete - Loss: {loss:.4f}, mIoU: {iou:.4f}")
    else:
        train_loader = DataLoader(ShapeNetPart(args.num_points, 'train'), batch_size=32, shuffle=True)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        for epoch in range(100):
            loss, iou = run_model(model, train_loader, criterion, device, optimizer)
            print(f"Epoch {epoch} - Loss: {loss:.4f}, mIoU: {iou:.4f}")
            if epoch % 10 == 0: torch.save(model.state_dict(), f'checkpoints/model_{epoch}.pth')