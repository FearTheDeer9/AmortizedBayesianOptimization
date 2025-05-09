"""Test the deprecated decorator utility function."""
import unittest
import warnings

from causal_meta.graph.utils import deprecated


class TestDeprecated(unittest.TestCase):
    """Test cases for the deprecated decorator."""

    def test_deprecated_warning(self):
        """Test that the deprecated decorator emits a warning."""
        # Create a test function decorated with @deprecated
        @deprecated(old_name="old_func", new_name="new_func")
        def test_func():
            return "test"

        # Check that calling the function emits a DeprecationWarning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = test_func()
            self.assertEqual(result, "test")  # Function should still work
            self.assertEqual(len(w), 1)  # Should have exactly one warning
            self.assertTrue(issubclass(w[0].category, DeprecationWarning))
            self.assertIn("'old_func' is deprecated", str(w[0].message))
            self.assertIn("Use 'new_func' instead", str(w[0].message))

    def test_deprecated_with_args(self):
        """Test that the deprecated decorator works with function arguments."""
        @deprecated(old_name="old_add", new_name="new_add")
        def add(a, b):
            return a + b

        # Check that the function works with arguments
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("ignore")
            result = add(1, 2)
            self.assertEqual(result, 3)

    def test_deprecated_with_kwargs(self):
        """Test that the deprecated decorator works with keyword arguments."""
        @deprecated(old_name="old_func", new_name="new_func")
        def test_func(a, b=2):
            return a + b

        # Check that the function works with keyword arguments
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("ignore")
            result = test_func(1, b=3)
            self.assertEqual(result, 4)

    def test_deprecated_preserves_metadata(self):
        """Test that the deprecated decorator preserves function metadata."""
        @deprecated(old_name="old_func", new_name="new_func")
        def test_func():
            """Test docstring."""
            return "test"

        # Check that functools.wraps preserved the docstring and name
        self.assertEqual(test_func.__name__, "test_func")
        self.assertEqual(test_func.__doc__, "Test docstring.")


if __name__ == "__main__":
    unittest.main() 