from dataclasses import dataclass, field
from typing import Callable


@dataclass
class GameState:
    """Tracks the state of a hangman game"""

    already_guessed: set[str] = field(default_factory=set)
    mistakes: int = 0
    max_mistakes: int = 7

    def clear(self):
        """Reset the game state for a new game"""
        self.already_guessed.clear()
        self.mistakes = 0

    def add_guess(self, letter: str):
        """Add a letter to already guessed set"""
        self.already_guessed.add(letter)

    def add_mistake(self):
        """Increment mistake counter"""
        self.mistakes += 1

    def is_game_over(self) -> bool:
        """Check if maximum mistakes reached"""
        return self.mistakes >= self.max_mistakes


def play_game(
    actual_word: str,
    guess_fn: Callable[
        [
            str,
        ],
        str,
    ],
    verbose: bool = False,
) -> int:
    """
    Play a hangman game with the given word and guessing function.

    Args:
        actual_word: The word to guess
        guess_fn: Function that takes current_word state and returns a letter guess

    Returns:
        1 if word was guessed successfully, 0 if failed
    """
    game_state = GameState()
    current_word = "".join(["_"] * len(actual_word))

    scrap_next = False

    while not game_state.is_game_over():
        guessed_letter = guess_fn(current_word, scrap=scrap_next)

        if verbose:
            print(f"{current_word}\n{guessed_letter}")

        # Validate guess
        if not guessed_letter.isalpha() or len(guessed_letter) != 1:
            raise ValueError("Guess must be a single letter")

        game_state.add_guess(guessed_letter)

        if guessed_letter in actual_word:
            # Reveal guessed letters in current_word
            new_word = list(current_word)
            for i, c in enumerate(actual_word):
                if c == guessed_letter:
                    new_word[i] = guessed_letter
            current_word = "".join(new_word)

            # Check if word is complete
            if current_word == actual_word:
                return 1
            scrap_next = False
        else:
            game_state.add_mistake()
            if current_word.count("_") <= 2:
                scrap_next = True

    return 0


def interactive_game(actual_word: str) -> int:
    """
    Play an interactive hangman game where user inputs guesses.

    Args:
        actual_word: The word to guess

    Returns:
        1 if word was guessed successfully, 0 if failed
    """

    def user_guess_fn(current_word: str) -> str:
        print(f"\nCurrent word: {current_word}")
        while True:
            guess = input("Enter your guess (single letter): ").lower().strip()
            if guess.isalpha() and len(guess) == 1:
                return guess
            print("Please enter a single letter.")

    return play_game(actual_word, user_guess_fn)


if __name__ == "__main__":
    # Example usage
    result = interactive_game("example")
    if result == 1:
        print("Congratulations! You've guessed the word.")
    else:
        print("Sorry, you've run out of guesses.")
