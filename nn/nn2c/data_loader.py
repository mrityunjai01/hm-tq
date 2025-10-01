import torch
import threading
import queue
import time

from .gen_functions import gen_x_y_for_word


class TrainBatchGenerator:
    def __init__(
        self,
        batch_size,
        words_file="w_train.txt",
        pos_embed_size=4,
        small_data=False,
        small_words=False,
        cache_size=10,  # Number of batches to cache
    ):
        self.batch_size = batch_size
        self.pos_embed_size = pos_embed_size
        self.cache_size = cache_size

        with open(words_file, "r") as f:
            words = [line.strip() for line in f.readlines()]
            if small_data:
                words = words[:2000]
            self.words_by_size = {}
            for word in words:
                if len(word) not in self.words_by_size:
                    self.words_by_size[len(word)] = []
                self.words_by_size[len(word)].append(word)
        if small_words:
            self.max_len = 5
        else:
            self.max_len = max(self.words_by_size.keys())

        self.curr_len = min(self.words_by_size.keys())
        self.curr_idx = 0

        # Threading setup
        self.batch_queue = queue.Queue(maxsize=cache_size)
        self.stop_event = threading.Event()
        self.producer_thread = threading.Thread(target=self._batch_producer, daemon=True)
        self.finished = False

        # Start the producer thread
        self.producer_thread.start()

    def _batch_producer(self):
        """Background thread that generates batches and puts them in the queue"""
        curr_len = min(self.words_by_size.keys())
        curr_idx = 0

        while not self.stop_event.is_set():
            try:
                # Check if queue is full before generating batch
                if self.batch_queue.full():
                    time.sleep(0.01)  # Wait a bit if queue is full
                    continue

                # Check if we've exhausted current length
                if curr_idx >= len(self.words_by_size[curr_len]):
                    if curr_len >= self.max_len:
                        # Signal end of data
                        self.batch_queue.put(None, timeout=1.0)
                        self.finished = True
                        break
                    else:
                        curr_len += 1
                        while (
                            curr_len not in self.words_by_size
                            or len(self.words_by_size[curr_len]) == 0
                        ) and (curr_len <= self.max_len):
                            curr_len += 1
                        if curr_len > self.max_len:
                            self.batch_queue.put(None, timeout=1.0)
                            self.finished = True
                            break
                        curr_idx = 0

                # Generate batch only if queue has space
                batch = self.words_by_size[curr_len][
                    curr_idx : min(
                        len(self.words_by_size[curr_len]), curr_idx + self.batch_size
                    )
                ]
                curr_idx += self.batch_size

                # Process batch
                data = [gen_x_y_for_word(word) for word in batch]
                x = torch.stack([d[0] for d in data])
                mask = torch.stack([d[1] for d in data])
                y = torch.stack([d[2] for d in data])

                # Put batch in queue (should not block since we checked queue is not full)
                self.batch_queue.put((x, mask, y), timeout=1.0)

            except queue.Full:
                # Queue became full between check and put, wait a bit
                time.sleep(0.01)
            except Exception as e:
                print(f"Error in batch producer: {e}")
                break

    def __iter__(self):
        return self

    def __next__(self):
        """Return next batch from cache"""
        try:
            batch = self.batch_queue.get(timeout=5.0)  # Wait up to 5 seconds
            if batch is None:  # End of data signal
                raise StopIteration
            return batch
        except queue.Empty:
            if self.finished:
                raise StopIteration
            else:
                raise RuntimeError("Timeout waiting for batch from producer thread")

    def stop(self):
        """Stop the producer thread and clean up resources"""
        self.stop_event.set()
        if self.producer_thread.is_alive():
            self.producer_thread.join(timeout=2.0)

    def __del__(self):
        """Cleanup when object is destroyed"""
        try:
            self.stop()
        except:
            pass  # Ignore errors during cleanup

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()
