from nn.nn2.train import train
import torch


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train(device=device, verbose=True)
