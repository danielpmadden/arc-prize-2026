"""Smoke tests for the supported installed entry point."""

import importlib
import unittest


class EntrypointTest(unittest.TestCase):
    def test_module_entrypoint_imports(self) -> None:
        module = importlib.import_module("renegade.__main__")
        self.assertTrue(callable(module.main))
