# ruff shouldn't  format this file

# fmt: off
import numpy as np
import pytest
from gen_functions import gen_x, gen_x_y_for_word


class TestGenX:
    def test_gen_x_simple_word(self):
        """Test gen_x with a simple revealed word pattern."""
        result = gen_x("cat", 3)
        expected = np.array([], 
                           dtype=np.int32)
        np.testing.assert_array_equal(result, expected)
    
    def test_gen_x_with_underscores(self):
        """Test gen_x with partially revealed word."""
        result = gen_x("c_t",3)
        expected = np.array([[26, 26, 2, 19, 26, 26]], 
                           dtype=np.int32)
        np.testing.assert_array_equal(result, expected)
    
    def test_gen_x_multiple_underscores(self):
        """Test gen_x with word longer than 17 characters."""
        result = gen_x('___', 3)
        expected = np.array([[26, 26, 26, 27, 27, 26],
                             [26, 26, 27, 27, 26, 26],
                             [26, 27, 27, 26, 26, 26],
                             ])
        np.testing.assert_array_equal(result, expected)
    


class TestGenXYForWord:
    def test_gen_x_y(self):
        """Test gen_x_y_for_word with a simple word."""
        # read words from w_test.txt
        pass

# fmt: on
if __name__ == "__main__":
    pytest.main([__file__])
