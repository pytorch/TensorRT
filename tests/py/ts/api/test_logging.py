import unittest

import torch_tensorrt as torchtrt


@unittest.skipIf(
    not torchtrt.ENABLED_FEATURES.torchscript_frontend,
    "TorchScript Frontend is not available",
)
class TestLoggingAPIs(unittest.TestCase):
    def test_logging_prefix(self):
        new_prefix = "Python API Test: "
        torchtrt.ts.logging.set_logging_prefix(new_prefix)
        logging_prefix = torchtrt.ts.logging.get_logging_prefix()
        self.assertEqual(new_prefix, logging_prefix)

    def test_reportable_log_level(self):
        new_level = torchtrt.ts.logging.Level.Error
        torchtrt.ts.logging.set_reportable_log_level(new_level)
        level = torchtrt.ts.logging.get_reportable_log_level()
        self.assertEqual(new_level, level)

    def test_is_colored_output_on(self):
        torchtrt.ts.logging.set_is_colored_output_on(True)
        color = torchtrt.ts.logging.get_is_colored_output_on()
        self.assertTrue(color)

    def test_context_managers(self):
        base_lvl = torchtrt.ts.logging.get_reportable_log_level()
        with torchtrt.logging.internal_errors():
            lvl = torchtrt.ts.logging.get_reportable_log_level()
            self.assertEqual(torchtrt.ts.logging.Level.InternalError, lvl)

        lvl = torchtrt.ts.logging.get_reportable_log_level()
        self.assertEqual(base_lvl, lvl)

        with torchtrt.logging.errors():
            lvl = torchtrt.ts.logging.get_reportable_log_level()
            self.assertEqual(torchtrt.ts.logging.Level.Error, lvl)

        lvl = torchtrt.ts.logging.get_reportable_log_level()
        self.assertEqual(base_lvl, lvl)

        with torchtrt.logging.warnings():
            lvl = torchtrt.ts.logging.get_reportable_log_level()
            self.assertEqual(torchtrt.ts.logging.Level.Warning, lvl)

        lvl = torchtrt.ts.logging.get_reportable_log_level()
        self.assertEqual(base_lvl, lvl)

        with torchtrt.logging.info():
            lvl = torchtrt.ts.logging.get_reportable_log_level()
            self.assertEqual(torchtrt.ts.logging.Level.Info, lvl)

        lvl = torchtrt.ts.logging.get_reportable_log_level()
        self.assertEqual(base_lvl, lvl)

        with torchtrt.logging.debug():
            lvl = torchtrt.ts.logging.get_reportable_log_level()
            self.assertEqual(torchtrt.ts.logging.Level.Debug, lvl)

        lvl = torchtrt.ts.logging.get_reportable_log_level()
        self.assertEqual(base_lvl, lvl)

        with torchtrt.logging.graphs():
            lvl = torchtrt.ts.logging.get_reportable_log_level()
            self.assertEqual(torchtrt.ts.logging.Level.Graph, lvl)

        lvl = torchtrt.ts.logging.get_reportable_log_level()
        self.assertEqual(base_lvl, lvl)


if __name__ == "__main__":
    unittest.main()
