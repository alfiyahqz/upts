import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import time
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time series data"""
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class EarlyStopping:
    """Early stopping to stop training when validation loss stops improving"""
    
    def __init__(self, patience=10, min_delta=0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = model.state_dict().copy()
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'   EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model = model.state_dict().copy()
            self.counter = 0


class Trainer:
    """Training manager for time series models"""
    
    def __init__(self, model, device='cuda', learning_rate=1e-3, 
                 batch_size=32, num_classes=3):
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_classes = num_classes
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # History
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch_X)
            loss = self.criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item() * batch_X.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = accuracy_score(all_labels, all_preds)
        
        return epoch_loss, epoch_acc
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Forward pass
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                # Statistics
                running_loss += loss.item() * batch_X.size(0)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
        
        epoch_loss = running_loss / len(val_loader.dataset)
        epoch_acc = accuracy_score(all_labels, all_preds)
        
        return epoch_loss, epoch_acc
    
    def fit(self, X_train, y_train, X_val, y_val, epochs=200, 
            patience=15, verbose=True):
        """
        Train the model
        
        Args:
            X_train: Training features (n_samples, seq_len, feature_dim)
            y_train: Training labels (n_samples,)
            X_val: Validation features
            y_val: Validation labels
            epochs: Maximum number of epochs
            patience: Early stopping patience
            verbose: Print progress
        
        Returns:
            history: Training history
        """
        # Create datasets and dataloaders
        train_dataset = TimeSeriesDataset(X_train, y_train)
        val_dataset = TimeSeriesDataset(X_val, y_val)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False
        )
        
        # Early stopping
        early_stopping = EarlyStopping(patience=patience, verbose=verbose)
        
        # Training loop
        print(f"\n🚀 Starting training...")
        print(f"   Device: {self.device}")
        print(f"   Batch size: {self.batch_size}")
        print(f"   Learning rate: {self.learning_rate}")
        print(f"   Max epochs: {epochs}")
        print(f"   Early stopping patience: {patience}")
        
        best_val_acc = 0.0
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Print progress
            if verbose and (epoch + 1) % 5 == 0:
                epoch_time = time.time() - epoch_start
                print(f"   Epoch {epoch+1:3d}/{epochs} | "
                      f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                      f"Time: {epoch_time:.1f}s")
            
            # Track best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
            
            # Early stopping
            early_stopping(val_loss, self.model)
            if early_stopping.early_stop:
                print(f"\n⏹️  Early stopping triggered at epoch {epoch+1}")
                # Restore best model
                self.model.load_state_dict(early_stopping.best_model)
                break
        
        total_time = time.time() - start_time
        print(f"\n✅ Training complete!")
        print(f"   Total time: {total_time/60:.1f} minutes")
        print(f"   Best validation accuracy: {best_val_acc:.4f}")
        
        return self.history
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model on test set
        
        Returns:
            metrics: Dictionary with accuracy, precision, recall, f1
            predictions: Model predictions
        """
        test_dataset = TimeSeriesDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        self.model.eval()
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(self.device)
                
                outputs = self.model(batch_X)
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_y.numpy())
                all_probs.extend(probs.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support = \
            precision_recall_fscore_support(
                all_labels, all_preds, average=None, zero_division=0
            )
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1_per_class': f1_per_class,
            'support': support,
            'confusion_matrix': cm
        }
        
        return metrics, all_preds, all_probs
    
    def save_model(self, path):
        """Save model checkpoint"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }, path)
        print(f"💾 Model saved to: {path}")
    
    def load_model(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        print(f"📂 Model loaded from: {path}")