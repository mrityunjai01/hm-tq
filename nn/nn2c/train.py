import torch
import tqdm
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
    small_data = True
    num_epochs = 4
    val_epoch_interval = 1
    initial_lr = 1e-3
    pos_weight_coeff = 10
    scheduler_tmax = 10
    early_stopping_patience = 5
    use_early_stopping = True
    test_word_count = 30
else:
    small_data = False
    num_epochs = 80
    val_epoch_interval = 1
    initial_lr = 1e-3
    pos_weight_coeff = 10
    early_stopping_patience = 30
    use_early_stopping = True
    test_word_count = 4000


def test_valid_shapes():
    for x, _, y in TrainBatchGenerator(
        batch_size=1024,
        small_data=small_data,
        words_file="hangman_data/train_words.txt",
    ):
        assert x.shape[0] == y.shape[0], (
            f"Batch dimensions don't match: x={x.shape[0]}, y={y.shape[0]}"
        )

        print(f"✓ Shapes are valid: x={x.shape}, y={y.shape}")
        break


def train(model_path="models/nn2c.pth", device=None, verbose=False):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HangmanNet(vocab_size=27, device=device, num_layers=4).to(device)
    if GPU:
        model = torch.compile(model, mode="max-autotune")
        scaler = GradScaler()
    print(f"{count_parameters(model)} params")
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=initial_lr, eps=5e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        5,
    )
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
            batch_size=64,
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
                loss.backward()

            if GPU:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

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

                # Check early stopping
                if early_stopping and early_stopping(winrate, model):
                    if verbose:
                        print(f"Early stopping at epoch {epoch + 1}")
                        print(f"Best winrate: {early_stopping.best_metric:.4f}")
                    break

            model.train()

    history.print_everything()


if __name__ == "__main__":
    test_valid_shapes()
    train(verbose=True)
