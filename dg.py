import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import argparse
import os
import json

def parse_args():
    parser = argparse.ArgumentParser(description='Domain Generalization with L2R Framework')
    
    parser.add_argument('--data_dir', type=str, default='/Users/mukher74/research/data',
                       help='Directory to store/load data')
    parser.add_argument('--num_domains', type=int, default=4,
                       help='Number of domains to generate')
    parser.add_argument('--num_classes', type=int, default=5,
                       help='Number of classes')
    parser.add_argument('--input_dim', type=int, default=20,
                       help='Input feature dimension')
    parser.add_argument('--hidden_dim', type=int, default=128,
                       help='Hidden layer dimension')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--save_model', action='store_true',
                       help='Save the trained model')
    parser.add_argument('--load_model', type=str, default=None,
                       help='Path to load a pre-trained model')
    parser.add_argument('--results_file', type=str, default='results.json',
                       help='File to save results')
    
    return parser.parse_args()

# --------------------------
# 1. Uncertainty Estimation
# --------------------------
class UncertaintyEstimator(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        
    def forward(self, x):
        logits, features = self.base_model(x)
        probs = F.softmax(logits, dim=1)
        
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
        max_prob = torch.max(probs, dim=1)[0]
        variance = torch.var(probs, dim=1)
        
        return {
            'logits': logits,
            'probs': probs,
            'entropy': entropy,
            'max_prob': max_prob,
            'variance': variance
        }

# ----------------------------
# 2. Calibration Networks
# ----------------------------
class DomainAwareCalibration(nn.Module):
    def __init__(self, num_domains, num_classes, hidden_dim=64):
        super().__init__()
        self.domain_embedding = nn.Embedding(num_domains, 8)
        self.calibration_net = nn.Sequential(
            nn.Linear(num_classes + 8, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, logits, domain_ids):
        domain_emb = self.domain_embedding(domain_ids)
        combined = torch.cat([logits, domain_emb], dim=1)
        calibration_adjustment = self.calibration_net(combined)
        calibrated_logits = logits + calibration_adjustment
        return calibrated_logits

class MetaCalibrationLearner(nn.Module):
    def __init__(self, base_calibrator, num_domains):
        super().__init__()
        self.base_calibrator = base_calibrator
        self.domain_discriminator = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_domains)
        )
        
    def forward(self, features, domain_ids):
        calibrated = self.base_calibrator(features, domain_ids)
        domain_pred = self.domain_discriminator(features)
        return calibrated, domain_pred

# ----------------------------
# 3. Rejection Mechanisms
# ----------------------------
class ContextAwareRejection(nn.Module):
    def __init__(self, feature_dim, hidden_dim=64):
        super().__init__()
        self.threshold_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, features):
        return self.threshold_net(features)

# ----------------------------
# 4. Domain Datasets
# ----------------------------
class DomainGeneralizationDataset(Dataset):
    def __init__(self, domains, num_classes, samples_per_domain=1000):
        self.domains = domains
        self.num_classes = num_classes
        self.data = []
        self.labels = []
        self.domain_ids = []
        
        for domain_id, domain in enumerate(domains):
            X, y = make_classification(
                n_samples=samples_per_domain,
                n_features=20,
                n_informative=15,
                n_redundant=5,
                n_classes=num_classes,
                class_sep=domain.get('class_sep', 2.0),
                flip_y=domain.get('flip_y', 0.1),
                shift=domain.get('shift', 0.0),
                scale=domain.get('scale', 1.0)
            )
            self.data.append(X)
            self.labels.append(y)
            self.domain_ids.extend([domain_id] * samples_per_domain)
        
        self.data = np.vstack(self.data)
        self.labels = np.concatenate(self.labels)
        self.domain_ids = np.array(self.domain_ids)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {
            'features': torch.tensor(self.data[idx], dtype=torch.float32),
            'label': torch.tensor(self.labels[idx], dtype=torch.long),
            'domain_id': torch.tensor(self.domain_ids[idx], dtype=torch.long)
        }

# ----------------------------
# 5. Core Models
# ----------------------------
class BaseModel(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=128):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits, features

# ----------------------------
# 6. Unified Framework
# ----------------------------
class UnifiedCalibratedL2R(nn.Module):
    def __init__(self, base_model, calibration_net, rejection_net, num_domains):
        super().__init__()
        self.base_model = base_model
        self.calibration_net = calibration_net
        self.rejection_net = rejection_net
        self.uncertainty_estimator = UncertaintyEstimator(base_model, num_domains)
        
    def forward(self, x, domain_id):
        logits, features = self.base_model(x)
        
        calibrated_logits = self.calibration_net(logits, domain_id)
        calibrated_probs = F.softmax(calibrated_logits, dim=1)
        
        uncertainty = self.uncertainty_estimator(x)['entropy']
        
        rejection_threshold = self.rejection_net(features).squeeze(-1)
        reject = uncertainty > rejection_threshold
        
        return {
            'logits': calibrated_logits,
            'probs': calibrated_probs,
            'uncertainty': uncertainty,
            'rejection_threshold': rejection_threshold,
            'reject': reject
        }
    
    def compute_loss(self, outputs, labels, domain_ids, alpha=0.5, beta=0.3):
        cls_loss = F.cross_entropy(outputs['logits'], labels)
        
        cal_loss = F.nll_loss(torch.log(outputs['probs'] + 1e-10), labels)
        
        correct = torch.argmax(outputs['probs'], dim=1) == labels
        rejection_loss = torch.mean(
            outputs['reject'].float() * correct.float() + 
            (1 - outputs['reject'].float()) * (1 - correct.float())
        )
        
        total_loss = cls_loss + alpha * cal_loss + beta * rejection_loss
        
        return total_loss

# ----------------------------
# 7. Training Utilities
# ----------------------------
def train_model(model, dataloader, optimizer, num_epochs, device):
    model.train()
    model.to(device)
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch in dataloader:
            features = batch['features'].to(device)
            labels = batch['label'].to(device)
            domain_ids = batch['domain_id'].to(device)
            
            optimizer.zero_grad()
            outputs = model(features, domain_ids)
            loss = model.compute_loss(outputs, labels, domain_ids)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader):.4f}")

# ----------------------------
# 8. Evaluation Metrics
# ----------------------------
def evaluate_model(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    rejected = 0
    ece = 0.0
    
    with torch.no_grad():
        for batch in dataloader:
            features = batch['features'].to(device)
            labels = batch['label'].to(device)
            domain_ids = batch['domain_id'].to(device)
            
            outputs = model(features, domain_ids)
            
            # Calculate accuracy on accepted samples
            accepted = ~outputs['reject']
            if torch.any(accepted):
                preds = torch.argmax(outputs['probs'][accepted], dim=1)
                correct += (preds == labels[accepted]).sum().item()
                total += accepted.sum().item()
            
            # Track rejection rate
            rejected += outputs['reject'].sum().item()
            
            # Calculate ECE (simplified)
            confidences, predictions = torch.max(outputs['probs'], dim=1)
            accuracy = (predictions == labels).float()
            ece += torch.abs(accuracy - confidences).mean().item()
    
    accuracy = correct / total if total > 0 else 0
    rejection_rate = rejected / len(dataloader.dataset)
    ece /= len(dataloader)
    
    return {
        'accuracy': accuracy,
        'rejection_rate': rejection_rate,
        'ece': ece
    }

# ----------------------------
# 9. Main Workflow
# ----------------------------
def print_usage():
    print("Domain Generalization with L2R Framework")
    print("=" * 50)
    print("Usage examples:")
    print("  python dg.py --epochs 50 --batch_size 128")
    print("  python dg.py --save_model --results_file my_results.json")
    print("  python dg.py --load_model trained_model.pth")
    print("\nAvailable arguments:")
    print("  --data_dir: Directory to store/load data (default: /Users/mukher74/research/data)")
    print("  --num_domains: Number of domains (default: 4)")
    print("  --num_classes: Number of classes (default: 5)")
    print("  --input_dim: Input feature dimension (default: 20)")
    print("  --hidden_dim: Hidden layer dimension (default: 128)")
    print("  --batch_size: Batch size (default: 64)")
    print("  --epochs: Number of training epochs (default: 20)")
    print("  --lr: Learning rate (default: 0.001)")
    print("  --save_model: Save the trained model")
    print("  --load_model: Path to load a pre-trained model")
    print("  --results_file: File to save results (default: results.json)")
    print("  --help: Show this help message")

def main():
    args = parse_args()
    
    if len(os.sys.argv) == 1:
        print_usage()
        return
    
    os.makedirs(args.data_dir, exist_ok=True)
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    domains = [
        {'name': 'Source', 'class_sep': 2.0, 'flip_y': 0.05},
        {'name': 'Easy Target', 'class_sep': 1.8, 'flip_y': 0.1},
        {'name': 'Hard Target', 'class_sep': 1.5, 'flip_y': 0.2},
        {'name': 'Shifted Target', 'shift': 1.0, 'scale': 1.2}
    ]
    
    train_dataset = DomainGeneralizationDataset(domains, args.num_classes)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    test_domains = [{'name': 'Severe Shift', 'shift': 1.5, 'scale': 1.5, 'flip_y': 0.3}]
    test_dataset = DomainGeneralizationDataset(test_domains, args.num_classes)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    base_model = BaseModel(args.input_dim, args.num_classes, args.hidden_dim)
    calibration_net = DomainAwareCalibration(args.num_domains, args.num_classes)
    rejection_net = ContextAwareRejection(args.hidden_dim)
    
    model = UnifiedCalibratedL2R(base_model, calibration_net, rejection_net, args.num_domains)
    
    if args.load_model:
        model_path = os.path.join(args.data_dir, args.load_model)
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            print(f"Loaded model from {model_path}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    train_model(model, train_loader, optimizer, args.epochs, DEVICE)
    
    if args.save_model:
        model_path = os.path.join(args.data_dir, 'trained_model.pth')
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
    
    train_metrics = evaluate_model(model, train_loader, DEVICE)
    test_metrics = evaluate_model(model, test_loader, DEVICE)
    
    results = {
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'args': vars(args)
    }
    
    results_path = os.path.join(args.data_dir, args.results_file)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    print(f"\nTraining Set Metrics:")
    print(f"Accuracy: {train_metrics['accuracy']:.4f}")
    print(f"Rejection Rate: {train_metrics['rejection_rate']:.4f}")
    print(f"ECE: {train_metrics['ece']:.4f}")
    
    print(f"\nTest Set Metrics:")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Rejection Rate: {test_metrics['rejection_rate']:.4f}")
    print(f"ECE: {test_metrics['ece']:.4f}")

if __name__ == "__main__":
    main()