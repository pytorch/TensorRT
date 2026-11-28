# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import importlib.util
import os
import platform
import unittest
from unittest import mock

from torch_tensorrt._utils import _cache_root


@unittest.skipUnless(platform.system() == "Linux", "uid lookup is Linux only")
class TestUidWithoutPasswdEntry(unittest.TestCase):
    def test_defaults_fall_back_to_the_uid(self):
        # Run the module code in a fresh module object so the real
        # torch_tensorrt.dynamo._defaults stays untouched.
        spec = importlib.util.find_spec("torch_tensorrt.dynamo._defaults")
        defaults = importlib.util.module_from_spec(spec)
        with mock.patch("pwd.getpwuid", side_effect=KeyError("uid not found")):
            spec.loader.exec_module(defaults)

        self.assertEqual(defaults.current_user, str(os.getuid()))

    def test_cache_root_falls_back_to_the_uid(self):
        for error in (KeyError, OSError):
            with (
                self.subTest(error=error),
                mock.patch("getpass.getuser", side_effect=error("uid not found")),
            ):
                self.assertEqual(_cache_root().name, f"torch_tensorrt_{os.getuid()}")


if __name__ == "__main__":
    unittest.main()
