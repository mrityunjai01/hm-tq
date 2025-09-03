import torch

from .gen_functions import gen_x_y_for_word


class TrainBatchGenerator:
    def __init__(
        self,
        batch_size,
        words_file="w_train.txt",
        pos_embed_size=4,
        small_data=False,
        surr=3,
    ):
        self.batch_size = batch_size
        self.surr = surr
        self.pos_embed_size = pos_embed_size
        with open(words_file, "r") as f:
            words = [line.strip() for line in f.readlines()]
            if small_data:
                words = words[:2000]
            self.words_by_size = {}
            for word in words:
                if len(word) not in self.words_by_size:
                    self.words_by_size[len(word)] = []
                self.words_by_size[len(word)].append(word)
        self.max_len = max(self.words_by_size.keys())

        self.curr_len = min(self.words_by_size.keys())
        self.curr_idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.curr_idx >= len(self.words_by_size[self.curr_len]):
            if self.curr_len >= self.max_len:
                raise StopIteration
            else:
                self.curr_len += 1
                while (
                    self.curr_len not in self.words_by_size
                    or len(self.words_by_size[self.curr_len]) == 0
                ) and (self.curr_len <= self.max_len):
                    self.curr_len += 1
                self.curr_idx = 0

        batch = self.words_by_size[self.curr_len][
            self.curr_idx : min(
                len(self.words_by_size[self.curr_len]), self.curr_idx + self.batch_size
            )
        ]
        self.curr_idx += self.batch_size

        data = [gen_x_y_for_word(word) for word in batch]
        x = torch.stack(
            [d[0] for d in data],
        )
        mask = torch.stack(
            [d[1] for d in data],
        )
        y = torch.stack([d[2] for d in data])

        return x, mask, y
