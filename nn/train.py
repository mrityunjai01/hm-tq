import os
import torch
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import hamming_loss, f1_score
from nn.model import create_model, MultiLabelLoss, count_parameters
from nn.data_loader import create_data_loaders


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience=10, min_delta=0.001, restore_best_weights=True):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            restore_best_weights: Whether to restore best weights on stop
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        """
        Check if training should stop.
        
        Args:
            val_loss: Current validation loss
            model: Model to potentially save weights from
            
        Returns:
            True if training should stop
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False


def train_model(
    model, 
    train_loader, 
    val_loader, 
    num_epochs=100,
    learning_rate=0.001,
    device='cpu',
    save_path='models/hangman_net.pth',
    use_early_stopping=True,
    patience=10
):
    """
    Train the neural network model.
    
    Args:
        model: HangmanNet model
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Maximum number of epochs
        learning_rate: Learning rate for optimizer
        device: Device to train on ('cpu' or 'cuda')
        save_path: Path to save the best model
        use_early_stopping: Whether to use early stopping
        patience: Early stopping patience
        
    Returns:
        Dictionary with training history
    """
    # Setup
    model = model.to(device)
    criterion = MultiLabelLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # Early stopping
    if use_early_stopping:
        early_stopping = EarlyStopping(patience=patience)
    
    # Logging
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    writer = SummaryWriter('runs/hangman_net')
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_hamming_loss': [],
        'val_f1_micro': [],
        'val_f1_macro': []
    }
    
    print(f"Training on {device}")
    print(f"Model parameters: {count_parameters(model):,}")
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_losses = []
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.6f}')
        
        # Validation phase
        model.eval()
        val_losses = []
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                loss = criterion(outputs, targets)
                
                val_losses.append(loss.item())
                
                # Convert to binary predictions for metrics
                predictions = (outputs.cpu().numpy() > 0.5).astype(int)
                all_predictions.append(predictions)
                all_targets.append(targets.cpu().numpy())
        
        # Calculate metrics
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        
        all_predictions = np.vstack(all_predictions)
        all_targets = np.vstack(all_targets)
        
        val_hamming = hamming_loss(all_targets, all_predictions)
        val_f1_micro = f1_score(all_targets, all_predictions, average='micro')
        val_f1_macro = f1_score(all_targets, all_predictions, average='macro')
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_hamming_loss'].append(val_hamming)
        history['val_f1_micro'].append(val_f1_micro)
        history['val_f1_macro'].append(val_f1_macro)
        
        # Logging
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Metrics/Hamming_Loss', val_hamming, epoch)
        writer.add_scalar('Metrics/F1_Micro', val_f1_micro, epoch)
        writer.add_scalar('Metrics/F1_Macro', val_f1_macro, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Print progress
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {train_loss:.6f}')
        print(f'  Val Loss: {val_loss:.6f}')
        print(f'  Val Hamming Loss: {val_hamming:.6f}')
        print(f'  Val F1 (micro): {val_f1_micro:.6f}')
        print(f'  Val F1 (macro): {val_f1_macro:.6f}')
        print(f'  LR: {optimizer.param_groups[0]["lr"]:.8f}')
        print('-' * 60)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'history': history
            }, save_path)
            print(f'New best model saved at epoch {epoch+1}')
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if use_early_stopping and early_stopping(val_loss, model):
            print(f'Early stopping at epoch {epoch+1}')
            break
    
    writer.close()
    return history


def main():
    """Main training function."""
    # Hyperparameters
    batch_size = 512
    learning_rate = 0.001
    num_epochs = 100
    hidden_dim1 = 512
    hidden_dim2 = 256
    dropout_rate = 0.3
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data loaders
    print("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(
        batch_size=batch_size, 
        test_split=0.2
    )
    
    # Model
    print("Creating model...")
    model = create_model(
        hidden_dim1=hidden_dim1,
        hidden_dim2=hidden_dim2,
        dropout_rate=dropout_rate
    )
    
    # Train
    print("Starting training...")
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        device=device,
        save_path='models/hangman_net.pth',
        use_early_stopping=True,
        patience=15
    )
    
    print("Training completed!")
    print(f"Best validation loss: {min(history['val_loss']):.6f}")
    print(f"Best F1 (micro): {max(history['val_f1_micro']):.6f}")
    print(f"Best F1 (macro): {max(history['val_f1_macro']):.6f}")


if __name__ == "__main__":
    main()