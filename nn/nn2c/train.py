import torch
import tqdm
import argparse
from torch import nn, optim
from stage import STAGE, GPU
from torch import autocast
from torch.amp.grad_scaler import GradScaler

from .data_loader import TrainBatchGenerator
from .history import History
from .model1 import HangmanNet, count_parameters
from .tst import test_model_on_game_play
from .early_stopping import EarlyStopping


surr = 4
if STAGE:
    batch_size = 64
    small_data = True
    num_epochs = 200
    val_epoch_interval = 1
    initial_lr = 1e-3
    pos_weight_coeff = 10
    scheduler_tmax = 10
    early_stopping_patience = 500
    use_early_stopping = True
    test_word_count = 100
else:
    batch_size = 64
    small_data = False
    num_epochs = 80
    val_epoch_interval = 1
    initial_lr = 1e-3
    pos_weight_coeff = 10
    early_stopping_patience = 30
    use_early_stopping = True
    test_word_count = 4000


def train(
    model_path="models/nn2c.pth",
    device=None,
    verbose=False,
    batch_size=None,
    scheduler_type="cosine",
    num_layers=4,
    hidden_dim=512,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Use passed batch_size or default from STAGE config
    if batch_size is None:
        batch_size = 64 if STAGE else 64

    model = HangmanNet(
        vocab_size=27, device=device, num_layers=num_layers, hidden_dim=hidden_dim
    ).to(device)

    if GPU:
        model = torch.compile(model, mode="max-autotune")
        scaler = GradScaler()

    print(f"{count_parameters(model)} params")
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=initial_lr, eps=5e-5)

    # Configure scheduler based on type
    if scheduler_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 5)
    elif scheduler_type == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)
    elif scheduler_type == "exponential":
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    elif scheduler_type == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=5, factor=0.7
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    history = History()
    best_val_metric = -1
    # Initialize early stopping
    early_stopping = None
    if use_early_stopping:
        early_stopping = EarlyStopping(
            patience=early_stopping_patience, min_delta=0.01, restore_best_weights=True
        )
    for epoch in tqdm.trange(num_epochs, disable=not verbose):
        total_loss = 0.0
        for x, mask, y in TrainBatchGenerator(
            batch_size=batch_size,
            small_data=small_data,
            words_file="hangman_data/train_words.txt",
        ):
            optimizer.zero_grad()
            x, mask, y = x.to(device), mask.to(device), y.to(device)

            if GPU:
                with autocast(device_type=device):
                    outputs = model(x)
                    loss = criterion(outputs[mask], y[mask])
                    scaler.scale(loss).backward()
            else:
                outputs = model(x)
                loss = criterion(outputs[mask], y[mask])
                # if math.isnan(loss.item()):
                #     breakpoint()
                loss.backward()

            if GPU:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            # Only step non-plateau schedulers here
            if scheduler_type != "plateau":
                scheduler.step()
            total_loss += loss.item()

        history.add_epoch(train_loss=total_loss)
        print(f"Trained epoch {epoch + 1}, Train Loss: {total_loss:.4f}")

        if epoch % val_epoch_interval == 0:
            model.eval()
            with torch.no_grad():
                winrate = test_model_on_game_play(
                    test_words_file="hangman_data/test_words.txt",
                    model_object=model,
                    max_test_words=test_word_count,
                )
                if winrate > best_val_metric:
                    best_val_metric = winrate
                    torch.save(model.state_dict(), f"{model_path}_checkpoint_{epoch}")
                history.add_epoch(val_metric=winrate)
                history.print_metrics()

                # Step plateau scheduler with validation metric
                # if scheduler_type == "plateau":
                #     scheduler.step(winrate)

                # Check early stopping
                if early_stopping and early_stopping(winrate, model):
                    if verbose:
                        print(f"Early stopping at epoch {epoch + 1}")
                        print(f"Best winrate: {early_stopping.best_metric:.4f}")
                    break

            model.train()

    history.print_everything()


def main():
    parser = argparse.ArgumentParser(description="Train HangmanNet model")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for training (default: 64)",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=["cosine", "step", "exponential", "plateau"],
        help="Learning rate scheduler type (default: cosine)",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=4,
        help="Number of transformer layers (default: 4)",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=512,
        help="Hidden dimension size (default: 512)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/nn2c.pth",
        help="Path to save model (default: models/nn2c.pth)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    train(
        model_path=args.model_path,
        verbose=args.verbose,
        batch_size=args.batch_size,
        scheduler_type=args.scheduler,
        num_layers=args.num_layers,
        hidden_dim=args.hidden_dim,
    )


if __name__ == "__main__":
    main()
