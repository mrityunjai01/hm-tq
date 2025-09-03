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
        
        self.best_metric = -float('inf')  # For winrate (higher is better)
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, metric, model):
        """
        Check if training should stop.
        
        Args:
            metric: Current validation metric (winrate)
            model: Model to potentially save weights from
            
        Returns:
            True if training should stop
        """
        if metric > self.best_metric + self.min_delta:
            self.best_metric = metric
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