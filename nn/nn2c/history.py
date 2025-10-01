import os
from dataclasses import dataclass, field


@dataclass
class History:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_metric: list[float] = field(default_factory=list)
    plot_directory: str = field(
        default_factory=lambda: os.path.join(os.path.dirname(__file__), "plots")
    )

    def add_epoch(
        self,
        train_loss: float | None = None,
        val_loss: float | None = None,
        val_metric: float | None = None,
    ):
        """Add metrics for a single epoch."""
        if train_loss is not None:
            self.train_loss.append(train_loss)
        if val_loss is not None:
            self.val_loss.append(val_loss)
        if val_metric is not None:
            self.val_metric.append(val_metric)

    def print_metrics(self):
        """Print the latest metrics."""
        if self.train_loss:
            print(f"Latest Train Loss: {self.train_loss[-1]}")
        if self.val_loss:
            print(f"Latest Validation Loss: {self.val_loss[-1]}")
        if self.val_metric:
            print(f"Latest Validation Metric: {self.val_metric[-1]}")

    def print_everything(self):
        """Print all recorded metrics."""
        print("Train Loss History:", self.train_loss)
        print("Validation Loss History:", self.val_loss)
        print("Validation Metric History:", self.val_metric)
