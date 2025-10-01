# ruff shouldn't  format this file

# fmt: off
import numpy as np
import pytest
from gen_functions import gen_x, gen_x_y_for_word


class TestGenX:
    def test_gen_x_simple_word(self):
        """Test gen_x with a simple revealed word pattern."""
        result = gen_x("cat")
        expected = np.array([[2, 0, 19, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26,
                            26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 2, 0, 19]], 
                           dtype=np.int32)
        np.testing.assert_array_equal(result, expected)
        assert result.shape == (1, 34)
    
    def test_gen_x_with_underscores(self):
        """Test gen_x with partially revealed word."""
        result = gen_x("c_t")
        expected = np.array([[2, 27, 19, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26,
                            26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 2, 27, 19]], 
                           dtype=np.int32)
        np.testing.assert_array_equal(result, expected)
    
    def test_gen_x_long_word(self):
        """Test gen_x with word longer than 17 characters."""
        long_word = "supercalifragilistic"  # 20 chars
        result = gen_x(long_word)
        assert result.shape == (1, 34)
    
    def test_gen_x_empty_string(self):
        """Test gen_x with empty string."""
        result = gen_x("")
        expected = 26 * np.ones((1, 34), dtype=np.int32)
        np.testing.assert_array_equal(result, expected)
    
    def test_gen_x_single_char(self):
        """Test gen_x with single character."""
        result = gen_x("a")
        expected = np.array([[0, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26,
                            26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 0]], 
                           dtype=np.int32)
        np.testing.assert_array_equal(result, expected)
    
    def test_gen_x_all_underscores(self):
        """Test gen_x with all underscores."""
        result = gen_x("____")
        expected = np.array([[27, 27, 27, 27, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26,
                            26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 27, 27, 27, 27]], 
                           dtype=np.int32)
        np.testing.assert_array_equal(result, expected)


class TestGenXYForWord:
    def test_gen_x_y(self):
        """Test gen_x_y_for_word with a simple word."""
        # read words from w_test.txt
        with open("w_test.txt", "r") as f:
            words = [line.strip() for line in f.readlines()]

        for word in words:
            x, y = gen_x_y_for_word(word)
            no_entries = x.shape[0]
            for i in range(no_entries):
                for elem in x[i]:
                    assert (0 <= elem <= 27)
                    if (elem != 26) and (elem != 27):

                        if (y[i][elem ] != 0):
                            print(f"Word: {word}, x: {x[i]}, y: {y[i]}")
                            assert(y[i][elem] == 0)

# fmt: on
if __name__ == "__main__":
    pytest.main([__file__])
