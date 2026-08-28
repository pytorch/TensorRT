import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch_tensorrt import _utils


class TestTensorRTLLMPlatformSupport(unittest.TestCase):
    @mock.patch.object(_utils.platform, "system", return_value="Linux")
    @mock.patch.object(_utils.platform, "machine", return_value="x86_64")
    @mock.patch.object(_utils.platform, "release", return_value="generic")
    @mock.patch.object(_utils, "is_thor", return_value=False)
    @mock.patch.object(_utils.trt, "__version__", "10.14.1")
    def test_cuda_12_is_supported(self, *unused_mocks):
        with mock.patch.object(torch.version, "cuda", "12.8"):
            self.assertTrue(_utils.is_platform_supported_for_trtllm())

    @mock.patch.object(_utils.platform, "system", return_value="Linux")
    @mock.patch.object(_utils.platform, "machine", return_value="x86_64")
    @mock.patch.object(_utils.platform, "release", return_value="generic")
    @mock.patch.object(_utils, "is_thor", return_value=False)
    @mock.patch.object(_utils.trt, "__version__", "10.14.1")
    def test_cuda_13_is_supported(self, *unused_mocks):
        with mock.patch.object(torch.version, "cuda", "13.0"):
            self.assertTrue(_utils.is_platform_supported_for_trtllm())

    @mock.patch.object(_utils.platform, "system", return_value="Linux")
    @mock.patch.object(_utils.platform, "machine", return_value="x86_64")
    @mock.patch.object(_utils.platform, "release", return_value="generic")
    @mock.patch.object(_utils, "is_thor", return_value=False)
    @mock.patch.object(_utils.trt, "__version__", "11.1.0")
    def test_incompatible_tensorrt_version_is_not_supported(self, *unused_mocks):
        with mock.patch.object(torch.version, "cuda", "13.0"):
            self.assertFalse(_utils.is_platform_supported_for_trtllm())

    @mock.patch.object(_utils.platform, "system", return_value="Linux")
    @mock.patch.object(_utils.platform, "machine", return_value="x86_64")
    @mock.patch.object(_utils.platform, "release", return_value="generic")
    @mock.patch.object(_utils, "is_thor", return_value=False)
    def test_cpu_only_pytorch_is_not_supported(self, mock_is_thor, *unused_mocks):
        with mock.patch.object(torch.version, "cuda", None):
            self.assertFalse(_utils.is_platform_supported_for_trtllm())
        mock_is_thor.assert_called_once_with()

    @mock.patch.object(_utils.platform, "system", return_value="Linux")
    @mock.patch.object(_utils.platform, "machine", return_value="x86_64")
    @mock.patch.object(_utils.platform, "release", return_value="generic")
    @mock.patch.object(_utils, "is_thor", return_value=False)
    def test_cuda_11_is_obsolete(self, *unused_mocks):
        with mock.patch.object(torch.version, "cuda", "11.8"):
            self.assertFalse(_utils.is_platform_supported_for_trtllm())


class TestTensorRTLLMVersion(unittest.TestCase):
    @mock.patch.object(_utils.platform, "system", return_value="Linux")
    @mock.patch.object(_utils.platform, "machine", return_value="x86_64")
    def test_cuda_13_auto_download_uses_cuda_13_artifact(self, *unused_mocks):
        with tempfile.TemporaryDirectory() as cache_root:
            plugin_path = (
                Path(cache_root)
                / "trtllm"
                / "1.2.0_linux_x86_64"
                / "tensorrt_llm"
                / "libs"
                / "libnvinfer_plugin_tensorrt_llm.so"
            )
            plugin_path.parent.mkdir(parents=True)
            plugin_path.touch()

            with mock.patch.object(
                _utils, "_cache_root", return_value=Path(cache_root)
            ):
                with mock.patch.object(torch.version, "cuda", "13.0"):
                    self.assertEqual(
                        _utils.download_and_get_plugin_lib_path(), str(plugin_path)
                    )


class TestTensorRTLLMLoading(unittest.TestCase):
    @mock.patch.object(_utils, "is_platform_supported_for_trtllm", return_value=True)
    @mock.patch.object(_utils, "load_and_initialize_trtllm_plugin", return_value=True)
    @mock.patch.object(_utils, "download_and_get_plugin_lib_path")
    def test_explicit_plugin_path_does_not_use_auto_download(
        self, mock_download, mock_load, *unused_mocks
    ):
        plugin_path = "/opt/trtllm/libnvinfer_plugin_tensorrt_llm.so"
        with mock.patch.dict(
            os.environ, {"TRTLLM_PLUGINS_PATH": plugin_path}, clear=True
        ):
            self.assertTrue(_utils.load_tensorrt_llm_for_nccl())

        mock_load.assert_called_once_with(plugin_path)
        mock_download.assert_not_called()

    @mock.patch.object(_utils, "is_platform_supported_for_trtllm", return_value=True)
    @mock.patch.object(_utils, "load_and_initialize_trtllm_plugin")
    @mock.patch.object(_utils, "download_and_get_plugin_lib_path", return_value=None)
    def test_auto_download_failure_does_not_try_to_load_none(
        self, mock_download, mock_load, *unused_mocks
    ):
        with mock.patch.dict(os.environ, {"USE_TRTLLM_PLUGINS": "1"}, clear=True):
            self.assertFalse(_utils.load_tensorrt_llm_for_nccl())

        mock_download.assert_called_once_with()
        mock_load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
