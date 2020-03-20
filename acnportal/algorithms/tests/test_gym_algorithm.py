# coding=utf-8
""" Tests for the ACN-Sim gym algorithm and model wrapper. """
import unittest
from importlib.util import find_spec

if find_spec("gym") is not None:
    from .. import SimRLModelWrapper


@unittest.skipIf(find_spec("gym") is None, "Requires gym install.")
class TestSimRLModelWrapper(unittest.TestCase):
    # noinspection PyMissingOrEmptyDocstring
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = object()
        cls.model_wrapper = SimRLModelWrapper(cls.model)

    def test_correct_on_init(self) -> None:
        self.assertEqual(self.model_wrapper.model, self.model)

    def test_predict_not_implemented_error(self) -> None:
        with self.assertRaises(NotImplementedError):
            self.model_wrapper.predict(*4*[None])


if __name__ == '__main__':
    unittest.main()
