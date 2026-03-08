import argparse, os, torch, time
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import DGCNN_PartSeg
from data import ShapeNetPart
from utils import calculate_shape_iou
from tqdm import tqdm  # <--- Ajout de tqdm pour la visibilité

def run_model(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    
    total_loss, total_iou = 0, 0
    
    # On entoure le loader avec tqdm pour voir l'avancement batch par batch
    mode = "🚀 Training" if is_train else "🔍 Eval"
    pbar = tqdm(enumerate(loader), total=len(loader), desc=mode, unit="batch", leave=False)
    print("📦 Préparation du premier batch (lecture des fichiers texte)...")
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
        
        # Mise à jour de la barre de progression avec la perte en temps réel
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / len(loader), total_iou / len(loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DGCNN Part Segmentation')
    parser.add_argument('--eval', action='store_true', help='Évaluer sans entraîner')
    parser.add_argument('--model_path', type=str, default='checkpoints/best_model.pth', help='Chemin du modèle')
    parser.add_argument('--num_points', type=int, default=1024, help='Nombre de points')
    parser.add_argument('--k', type=int, default=20, help='Voisins pour EdgeConv')
    parser.add_argument('--batch_size', type=int, default=16, help='Taille du batch')
    parser.add_argument('--epochs', type=int, default=50, help='Nombre d\'époques')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    args = parser.parse_args()

    if not os.path.exists('checkpoints'): os.makedirs('checkpoints')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DGCNN_PartSeg(k=args.k).to(device)
    criterion = nn.CrossEntropyLoss()

    if args.eval:
        print("\n🔍 --- MODE ÉVALUATION ---")
        model.load_state_dict(torch.load(args.model_path))
        # Optimisation aussi pour l'évaluation
        test_loader = DataLoader(
            ShapeNetPart(args.num_points, 'test'), 
            batch_size=args.batch_size,
            num_workers=0,
            pin_memory=True
        )
        loss, iou = run_model(model, test_loader, criterion, device)
        print(f"📊 Résultats Test | Loss: {loss:.4f} | mIoU: {iou:.4f}\n")
    else:
        print("\n" + "="*60)
        print("🚀 LANCEMENT DE L'ENTRAÎNEMENT DGCNN (ShapeNet Part Seg)")
        print(f"💻 Matériel : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
        print(f"📦 Config : Points={args.num_points}, Batch={args.batch_size}, Workers=4")
        print("="*60 + "\n")

        # --- DATALOADER BOOSTÉ ---
        train_loader = DataLoader(
            ShapeNetPart(args.num_points, 'train'), 
            batch_size=args.batch_size, 
            shuffle=True,
            num_workers=4,           # Passé à 4 pour paralléliser la lecture
            pin_memory=True,         # Indispensable pour accélérer le GPU
            prefetch_factor=2,       # Prépare les batchs pendant que le GPU calcule
            persistent_workers=True   # Garde les processus en vie entre les époques
        )
        
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        best_iou = 0
        start_time = time.time()

        for epoch in range(args.epochs):
            epoch_start = time.time()
            loss, iou = run_model(model, train_loader, criterion, device, optimizer)
            epoch_duration = time.time() - epoch_start
            
            is_best = iou > best_iou
            status = "⭐ NEW BEST" if is_best else "➖"
            
            # Ce print s'affichera APRÈS la barre de progression de l'époque
            print(f"📅 Epoch [{epoch+1:03d}/{args.epochs}] | Loss: {loss:.4f} | mIoU: {iou:.4f} | ⏱️ {epoch_duration:.1f}s | {status}")

            if is_best:
                best_iou = iou
                torch.save(model.state_dict(), 'checkpoints/best_model.pth')

        total_duration = (time.time() - start_time) / 60
        print("\n" + "="*60)
        print(f"✅ Terminé en {total_duration:.2f} min | Meilleur mIoU: {best_iou:.4f}")
        print("="*60)