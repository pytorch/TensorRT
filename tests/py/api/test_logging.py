import unittest
import torch_tensorrt as torchtrt
import torch
import torchvision.models as models
import copy
from typing import Dict


class TestLoggingAPIs(unittest.TestCase):
    def test_logging_prefix(self):
        new_prefix = "Python API Test: "
        torchtrt.logging.set_logging_prefix(new_prefix)
        logging_prefix = torchtrt.logging.get_logging_prefix()
        self.assertEqual(new_prefix, logging_prefix)

    def test_reportable_log_level(self):
        new_level = torchtrt.logging.Level.Error
        torchtrt.logging.set_reportable_log_level(new_level)
        level = torchtrt.logging.get_reportable_log_level()
        self.assertEqual(new_level, level)

    def test_is_colored_output_on(self):
        torchtrt.logging.set_is_colored_output_on(True)
        color = torchtrt.logging.get_is_colored_output_on()
        self.assertTrue(color)

    def test_context_managers(self):
        base_lvl = torchtrt.logging.get_reportable_log_level()
        with torchtrt.logging.internal_errors():
            lvl = torchtrt.logging.get_reportable_log_level()
            self.assertEqual(torchtrt.logging.Level.InternalError, lvl)

        lvl = torchtrt.logging.get_reportable_log_level()
        self.assertEqual(base_lvl, lvl)

        with torchtrt.logging.errors():
            lvl = torchtrt.logging.get_reportable_log_level()
            self.assertEqual(torchtrt.logging.Level.Error, lvl)

        lvl = torchtrt.logging.get_reportable_log_level()
        self.assertEqual(base_lvl, lvl)

        with torchtrt.logging.warnings():
            lvl = torchtrt.logging.get_reportable_log_level()
            self.assertEqual(torchtrt.logging.Level.Warning, lvl)

        lvl = torchtrt.logging.get_reportable_log_level()
        self.assertEqual(base_lvl, lvl)

        with torchtrt.logging.info():
            lvl = torchtrt.logging.get_reportable_log_level()
            self.assertEqual(torchtrt.logging.Level.Info, lvl)

        lvl = torchtrt.logging.get_reportable_log_level()
        self.assertEqual(base_lvl, lvl)

        with torchtrt.logging.debug():
            lvl = torchtrt.logging.get_reportable_log_level()
            self.assertEqual(torchtrt.logging.Level.Debug, lvl)

        lvl = torchtrt.logging.get_reportable_log_level()
        self.assertEqual(base_lvl, lvl)

        with torchtrt.logging.graphs():
            lvl = torchtrt.logging.get_reportable_log_level()
            self.assertEqual(torchtrt.logging.Level.Graph, lvl)

        lvl = torchtrt.logging.get_reportable_log_level()
        self.assertEqual(base_lvl, lvl)


if __name__ == "__main__":
    unittest.main()
