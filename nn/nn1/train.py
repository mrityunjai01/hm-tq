import torch
import tqdm
from torch import nn, optim
from stage import STAGE

from nn.nn1.data_loader import TrainBatchGenerator
from nn.nn1.history import History
from nn.nn1.model1 import HangmanNet
from nn.nn1.tst import test_model_on_game_play


def test_valid_shapes():
    for x, y in TrainBatchGenerator(batch_size=64):
        # Assert batch dimensions match
        assert x.shape[0] == y.shape[0], (
            f"Batch dimensions don't match: x={x.shape[0]}, y={y.shape[0]}"
        )

        # Assert feature dimensions are correct
        assert x.shape[1] == 34, f"x second dimension should be 34, got {x.shape[1]}"
        assert y.shape[1] == 26, f"y second dimension should be 26, got {y.shape[1]}"

        print(f"✓ Shapes are valid: x={x.shape}, y={y.shape}")
        break


if STAGE:
    small_data = True
    num_epochs = 2
    val_epoch_interval = 1
    save_interval = 1
    initial_lr = 1e-3
    pos_weight_coeff = 10
    scheduler_tmax = 10
else:
    small_data = False
    num_epochs = 20
    val_epoch_interval = 1
    save_interval = 5
    initial_lr = 1e-3
    pos_weight_coeff = 10
    scheduler_tmax = 10


def train(model_path="models/nn1.pth", device=None, verbose=False):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HangmanNet().to(device)
    pos_weight = torch.ones(26) * pos_weight_coeff

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=scheduler_tmax)
    history = History()

    for epoch in tqdm.trange(num_epochs, disable=not verbose):
        total_loss = 0.0
        for x, y in TrainBatchGenerator(batch_size=64, small_data=small_data):
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y.float())
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        history.add_epoch(train_loss=total_loss)

        model.eval()
        if epoch % val_epoch_interval == 0:
            winrate = test_model_on_game_play(model_object=model)
            history.add_epoch(val_metric=winrate)
        model.train()

        if epoch % save_interval == 0:
            torch.save(model.state_dict(), model_path)

    history.print_metrics()


if __name__ == "__main__":
    test_valid_shapes()
    train(verbose=True)
