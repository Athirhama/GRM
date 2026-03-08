import argparse, os, torch, time
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import DGCNN_PartSeg
from data import ShapeNetPart
from utils import calculate_shape_iou

def run_model(model, loader, criterion, device, optimizer=None):
    """
    Exécute une passe complète (entraînement ou validation) sur le dataset.
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    
    total_loss, total_iou = 0, 0
    
    for data, label, seg in loader:
        # Envoi des données sur le GPU (Tesla T4 sur Colab)
        data, label, seg = data.to(device), label.to(device), seg.to(device)
        batch_size = data.size(0)
        
        # Vecteur One-hot pour injecter la catégorie de l'objet dans le décodeur
        l_one_hot = torch.zeros(batch_size, 16).to(device)
        l_one_hot.scatter_(1, label.view(-1, 1), 1)

        if is_train: 
            optimizer.zero_grad() # Reset des gradients pour éviter l'accumulation
        
        with torch.set_grad_enabled(is_train):
            # Forward pass : calcul des logits (prédictions non normalisées)
            logits = model(data, l_one_hot)
            # Loss CrossEntropy sur tous les points du batch
            loss = criterion(logits.view(-1, 50), seg.view(-1))
            
            if is_train:
                loss.backward() # Calcul de l'erreur
                optimizer.step() # Mise à jour des poids du DGCNN
        
        total_loss += loss.item()
        # La classe prédite est l'indice avec la valeur maximale
        preds = logits.max(dim=1)[1]
        # Calcul du mIoU (Moyenne de l'Intersection over Union)
        total_iou += calculate_shape_iou(preds, seg)

    return total_loss / len(loader), total_iou / len(loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DGCNN Part Segmentation')
    parser.add_argument('--eval', action='store_true', help='Évaluer sans entraîner')
    parser.add_argument('--model_path', type=str, default='checkpoints/best_model.pth', help='Chemin du modèle')
    parser.add_argument('--num_points', type=int, default=1024, help='Nombre de points (1024 par défaut)')
    parser.add_argument('--k', type=int, default=20, help='Voisins pour le graphe EdgeConv')
    parser.add_argument('--batch_size', type=int, default=16, help='Taille du batch (16 conseillé sur Colab)')
    parser.add_argument('--epochs', type=int, default=50, help='Nombre d\'époques')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    args = parser.parse_args()

    if not os.path.exists('checkpoints'): os.makedirs('checkpoints')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialisation du modèle avec le paramètre k pour la convolution de graphe
    model = DGCNN_PartSeg(k=args.k).to(device)
    criterion = nn.CrossEntropyLoss()

    if args.eval:
        print("\n🔍 --- MODE ÉVALUATION ---")
        model.load_state_dict(torch.load(args.model_path))
        test_loader = DataLoader(ShapeNetPart(args.num_points, 'test'), batch_size=args.batch_size)
        loss, iou = run_model(model, test_loader, criterion, device)
        print(f"📊 Résultats Test | Loss: {loss:.4f} | mIoU: {iou:.4f}\n")
    else:
        print("\n" + "="*60)
        print("🚀 LANCEMENT DE L'ENTRAÎNEMENT DGCNN (ShapeNet Part Seg)")
        print(f"💻 Matériel détecté : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
        print(f"📦 Config : Points={args.num_points}, k={args.k}, Batch={args.batch_size}")
        print("="*60 + "\n")

        train_loader = DataLoader(
          ShapeNetPart(args.num_points, 'train'), 
          batch_size=args.batch_size, 
          shuffle=True,
          num_workers=2,      # Utilise 2 coeurs CPU pour lire les fichiers en avance
          pin_memory=True )   # Accélère le transfert CPU -> GPU
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        
        best_iou = 0
        start_time = time.time()

        for epoch in range(args.epochs):
            epoch_start = time.time()
            
            # Exécution de l'époque
            loss, iou = run_model(model, train_loader, criterion, device, optimizer)
            
            epoch_duration = time.time() - epoch_start
            
            # Log visuel de progression
            is_best = iou > best_iou
            status = "⭐ NEW BEST" if is_best else "➖"
            
            print(f"📅 Époque [{epoch+1:03d}/{args.epochs}] | 📉 Loss: {loss:.6f} | 🎯 mIoU: {iou:.4f} | ⏱️ {epoch_duration:.1f}s | {status}")

            if is_best:
                best_iou = iou
                torch.save(model.state_dict(), 'checkpoints/best_model.pth')

        total_duration = (time.time() - start_time) / 60
        print("\n" + "="*60)
        print(f"✅ Terminé en {total_duration:.2f} min | Meilleur mIoU: {best_iou:.4f}")
        print("="*60)