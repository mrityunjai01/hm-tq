import os
import matplotlib.pyplot as plt
from dataclasses import dataclass, field


@dataclass
class History:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_metric: list[float] = field(default_factory=list)
    plot_directory: str = field(
        default_factory=lambda: os.path.join(os.path.dirname(__file__), "plots")
    )

    def plot(self):
        """Plot training history and save to files."""
        os.makedirs(self.plot_directory, exist_ok=True)

        if self.train_loss or self.val_loss:
            plt.figure(figsize=(10, 6))
            if self.train_loss:
                plt.plot(self.train_loss, label="Train Loss", color="blue")
            if self.val_loss:
                plt.plot(self.val_loss, label="Validation Loss", color="red")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Training and Validation Loss")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(self.plot_directory, "loss_plot.png"))
            plt.close()

        if self.val_metric:
            plt.figure(figsize=(10, 6))
            plt.plot(self.val_metric, label="Validation Metric", color="green")
            plt.xlabel("Epoch")
            plt.ylabel("Metric")
            plt.title("Validation Metric")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(self.plot_directory, "metric_plot.png"))
            plt.close()

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
