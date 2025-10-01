import torch
import tqdm
from torch import nn, optim
from stage import STAGE

from .data_loader import TrainBatchGenerator
from .history import History
from .model1 import HangmanNet
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
else:
    small_data = False
    num_epochs = 20
    val_epoch_interval = 1
    initial_lr = 1e-3
    pos_weight_coeff = 10
    scheduler_tmax = 10
    early_stopping_patience = 10
    use_early_stopping = True


def test_valid_shapes():
    for x, y in TrainBatchGenerator(batch_size=64, small_data=small_data, surr=surr):
        # Assert batch dimensions match
        assert x.shape[0] == y.shape[0], (
            f"Batch dimensions don't match: x={x.shape[0]}, y={y.shape[0]}"
        )

        # Assert feature dimensions are correct
        assert x.shape[1] == 2 * surr, (
            f"x second dimension should be {2 * surr}, got {x.shape[1]}"
        )

        print(f"✓ Shapes are valid: x={x.shape}, y={y.shape}")
        break


def train(model_path="models/nn2a_nopos.pth", device=None, verbose=False):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HangmanNet(
        input_dim=2 * surr, vocab_size=27, target_vocab_size=26, device=device
    ).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=scheduler_tmax)
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
        for x, y in TrainBatchGenerator(
            batch_size=64, small_data=small_data, surr=surr
        ):
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        history.add_epoch(train_loss=total_loss)

        if epoch % val_epoch_interval == 0:
            model.reset_last_embedding()
            model.eval()
            with torch.no_grad():
                winrate = test_model_on_game_play(model_object=model, surr=surr)
                if winrate > best_val_metric:
                    best_val_metric = winrate
                    torch.save(model.state_dict(), model_path)
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
