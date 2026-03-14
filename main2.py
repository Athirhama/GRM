import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR 
from model2 import DGCNN_PartSeg
from data2 import ShapeNetPart
from utils import calculate_shape_iou
from tqdm import tqdm 

def run_model(model, loader, criterion, device, optimizer=None):
    """
    Fonction unique pour l'entraînement et l'évaluation.
    Si un optimiseur est fourni, on entraîne. Sinon, on évalue.
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss, total_iou = 0.0, 0.0
    
    mode = "Train" if is_train else "Test "
    pbar = tqdm(enumerate(loader), total=len(loader), desc=mode, unit="batch", leave=False)
    
    for i, (data, label, seg) in pbar:
        data, label, seg = data.to(device), label.to(device), seg.to(device)
        batch_size = data.size(0)
        
        # Création du tenseur one-hot pour la catégorie
        l_one_hot = torch.zeros(batch_size, 16).to(device)
        l_one_hot.scatter_(1, label.view(-1, 1), 1)

        if is_train: 
            optimizer.zero_grad()
        
        with torch.set_grad_enabled(is_train):
            logits = model(data, l_one_hot)
            # 50 correspond au nombre total de parties (labels) possibles dans ShapeNetPart
            loss = criterion(logits.view(-1, 50), seg.view(-1))
            
            if is_train:
                loss.backward()
                optimizer.step()
        
        total_loss += loss.item()
        preds = logits.max(dim=1)[1]
        
        # Assure-toi que ta fonction calculate_shape_iou gère bien les batchs
        total_iou += calculate_shape_iou(preds, seg)
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / len(loader), total_iou / len(loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DGCNN Part Segmentation')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size (Papier: 32)')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs (Papier: 200)')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate (Papier: 0.1)')
    # Le papier indique 2048 points pour la tâche de Part Segmentation
    parser.add_argument('--num_points', type=int, default=2048, help='Number of points')
    parser.add_argument('--k', type=int, default=20, help='Number of nearest neighbors')
    parser.add_argument('--eval', action='store_true', help='Run evaluation mode only')
    parser.add_argument('--model_path', type=str, default='checkpoints/best_model.pth', help='Path to load weights')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DGCNN_PartSeg(k=args.k).to(device)
    
    # Support Multi-GPU si disponible (mentionné dans le papier)
    if torch.cuda.device_count() > 1:
        print(f"Utilisation de {torch.cuda.device_count()} GPUs !")
        model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()

    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')

    if args.eval:
        print(f"Loading checkpoint: {args.model_path}")
        if os.path.exists(args.model_path):
            # Ajout de map_location pour éviter les bugs si entraîné sur GPU et testé sur CPU
            model.load_state_dict(torch.load(args.model_path, map_location=device))
            
            test_loader = DataLoader(
                ShapeNetPart(args.num_points, 'test'), 
                batch_size=args.batch_size, shuffle=False, num_workers=4
            )
            
            test_loss, test_iou = run_model(model, test_loader, criterion, device)
            print(f"\nFinal Test Results | Loss: {test_loss:.4f} | mIoU: {test_iou:.4f}")
        else:
            print("Error: No checkpoint found at the specified path.")

    else:
        # Optimiseur SGD avec Momentum (0.9)
        # Le weight_decay=1e-4 est une excellente pratique de régularisation (L2 penalty)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
        
        # Cosine Annealing : réduit le LR de 0.1 jusqu'à 0.001
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.001) 
        
        train_loader = DataLoader(
            ShapeNetPart(args.num_points, 'train'), 
            batch_size=args.batch_size, 
            shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True
        )
        
        # NOUVEAU : On prépare le test_loader pour évaluer à la volée
        test_loader = DataLoader(
            ShapeNetPart(args.num_points, 'test'), 
            batch_size=args.batch_size, 
            shuffle=False, num_workers=4, pin_memory=True
        )

        best_iou = 0.0
        print("Démarrage de l'entraînement...")
        for epoch in range(args.epochs):
            # 1. Phase d'entraînement
            train_loss, train_iou = run_model(model, train_loader, criterion, device, optimizer=optimizer)
            scheduler.step() 
            current_lr = optimizer.param_groups[0]['lr']
            
            # 2. Phase d'évaluation (Crucial pour sauvegarder le bon modèle)
            test_loss, test_iou = run_model(model, test_loader, criterion, device, optimizer=None)
            
            # 3. Sauvegarde basée sur le score de TEST
            is_best = test_iou > best_iou
            if is_best:
                best_iou = test_iou
                # Si DataParallel est utilisé, on sauvegarde model.module.state_dict()
                state_to_save = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
                torch.save(state_to_save, 'checkpoints/best_model.pth')
                
            print(f"Epoch [{epoch+1:03d}/{args.epochs}] | LR: {current_lr:.5f}")
            print(f"  Train -> Loss: {train_loss:.4f} | mIoU: {train_iou:.4f}")
            print(f"  Test  -> Loss: {test_loss:.4f} | mIoU: {test_iou:.4f} {'[BEST]' if is_best else ''}")
