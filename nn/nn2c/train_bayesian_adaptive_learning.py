import torch
import tqdm
import argparse
import optuna
from optuna.samplers import TPESampler
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


def train_single_trial(
    batch_size,
    scheduler_type,
    num_layers,
    hidden_dim,
    learning_rate,
    dropout_rate,
    device=None,
    verbose=False,
    trial_epochs=10,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = HangmanNet(
        vocab_size=27,
        device=device,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        dropout_rate=dropout_rate,
    ).to(device)

    if GPU:
        model = torch.compile(model, mode="max-autotune")
        scaler = GradScaler()

    if verbose:
        print(f"{count_parameters(model)} params")
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, eps=5e-5)

    # Configure scheduler based on type
    if scheduler_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 6)
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

    for epoch in tqdm.trange(trial_epochs, disable=not verbose):
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
        if verbose:
            print(f"Trained epoch {epoch + 1}, Train Loss: {total_loss:.4f}")

    # Final validation
    model.eval()
    with torch.no_grad():
        final_winrate = test_model_on_game_play(
            test_words_file="hangman_data/test_words.txt",
            model_object=model,
            max_test_words=1000,  # Smaller test set for faster trials
            verbose=False,
        )
        history.add_epoch(val_metric=final_winrate)

    return final_winrate


def objective(trial):
    """Optuna objective function for hyperparameter optimization."""
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    scheduler_type = trial.suggest_categorical(
        "scheduler_type", ["cosine", "step", "exponential", "plateau"]
    )
    num_layers = trial.suggest_int("num_layers", 2, 6)
    hidden_dim = trial.suggest_categorical("hidden_dim", [256, 512, 768, 1024])
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)

    try:
        final_winrate = train_single_trial(
            batch_size=batch_size,
            scheduler_type=scheduler_type,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            learning_rate=learning_rate,
            dropout_rate=dropout_rate,
            trial_epochs=10,
            verbose=False,
        )
        return final_winrate
    except Exception as e:
        print(f"Trial failed with error: {e}")
        return 0.0  # Return poor score for failed trials


def main():
    parser = argparse.ArgumentParser(
        description="Bayesian hyperparameter optimization for HangmanNet"
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Number of optimization trials (default: 100)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=8,
        help="Number of parallel jobs (default: 8)",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default="hangman_optuna_study",
        help="Name of the Optuna study (default: hangman_optuna_study)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    study = optuna.create_study(
        direction="maximize", sampler=TPESampler(seed=42), study_name=args.study_name
    )

    print(
        f"Starting Bayesian optimization with {args.n_trials} trials using {args.n_jobs} parallel jobs"
    )

    # Optimize with parallel execution
    study.optimize(
        objective, n_trials=args.n_trials, n_jobs=args.n_jobs, show_progress_bar=True
    )

    # Print results
    print("\n=== Optimization Results ===")
    print(f"Best trial value: {study.best_value:.4f}")
    print(f"Best hyperparameters: {study.best_params}")

    # Save results
    best_params_file = f"best_params_{args.study_name}.txt"
    with open(best_params_file, "w") as f:
        f.write(f"Best validation winrate: {study.best_value:.4f}\n")
        f.write("Best hyperparameters:\n")
        for key, value in study.best_params.items():
            f.write(f"  {key}: {value}\n")

    print(f"Best parameters saved to {best_params_file}")

    # Optionally show importance of hyperparameters
    try:
        importance = optuna.importance.get_param_importances(study)
        print("\nHyperparameter importance:")
        for param, imp in importance.items():
            print(f"  {param}: {imp:.4f}")
    except:
        print("Could not compute parameter importance (insufficient trials)")

    return study


if __name__ == "__main__":
    main()
