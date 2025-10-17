# TQ HM Challenge

This notebook describes code to play a wrod guessing game. The game is similar to wordle, you have to guess a letter which could take the place of '\_ ' in a sequence of letters and underscores.

- Say, you're given a word "c_t" (exclude the quotes from consideration ), you might guess it to be "cat", "cot", "cut", etc. formed from replacing '\*' with a letter.
- You have six max incorrect tries at guessing the letters, each time you guess a letter correctly you have made incremental progress toward guessing all the letters and ending the game.

> '\_\_\_'

> I guess 'l'

> '\_\_\_'

> I guess 'c'

> 'c\_\_'

> I guess 't'

> 'c_t'

> I guess 'e'

> 'c_t'

> I guess 'p'

> 'c_t'

> I guess 'r'

> 'c_t'

> I guess 'q'

> 'c_t', game ends at this because of six incorrect tries.

# Components

Most components are similar across the different folders within `nn/`, I will take the example of `nn/nn3c` when talking about files.

- Model: `model1.py`
- Tuning: `bayesian_adaptive_learning.py`
- Training and Running: `runner.py`, `train.py`, `train_small.py`
- Data processing: `data_loader.py` `combinations.py` `gen_functions.py`
- Prediction: `predict.py`, `hconfig.py`, `scrap.py`, `vowels.py`, `selector.py`

# Key Things

- Generating words with blanks that resemble the real test time tasks. Geometric distribution over the replacement of letters by '\_'.
- Model is just a decoder with a RoPE encoding. Better capture spatial relationships in the words.
- Selector (`selector`) generates data for selection among the different prediction techniques using a shallow neural net.
- Prediction uses multiple techniques to predict the next letter. The selector chooses among them.
- Tuning across configurations to find the best trained model at the end of 10 epochs (the term 'epoch' is not well defined because the data used in every pass changes)

# Usage

Have a look through the 'requirements.txt' file and the 'nn/nn2c/runner.py' python file.

```python
pip install -r requirements.txt
```

```python
python -m nn.nn2c.runner
```
