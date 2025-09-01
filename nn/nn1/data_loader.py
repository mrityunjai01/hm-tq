import numpy as np
import torch

from nn.nn1.gen_functions import gen_x, gen_x_y_for_word


class TrainBatchGenerator:
    def __init__(self, batch_size, words_file="w_train.txt", small_data=False):
        self.batch_size = batch_size
        with open(words_file, "r") as f:
            self.words = [line.strip() for line in f.readlines()]
        if small_data:
            self.words = self.words[:1000]
        self.curr_idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.curr_idx >= len(self.words):
            raise StopIteration
        batch = self.words[
            self.curr_idx : min(len(self.words), self.curr_idx + self.batch_size)
        ]
        self.curr_idx += self.batch_size
        data = [gen_x_y_for_word(word) for word in batch]
        x_batch = np.concatenate([d[0] for d in data], axis=0)
        y_batch = np.concatenate([d[1] for d in data], axis=0)
        return torch.from_numpy(x_batch), torch.from_numpy(y_batch)
