%bash

git config --global --add safe.directory /home/TensorRT

#Install bazel
apt install apt-transport-https curl gnupg -y
curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor >bazel-archive-keyring.gpg
mv bazel-archive-keyring.gpg /usr/share/keyrings
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/bazel-archive-keyring.gpg] https://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list


apt update && apt install bazel-8.1.1
apt install bazel
bazel
cd /home/TensorRT

python -m pip install --pre -e . --extra-index-url https://download.pytorch.org/whl/nightly/cu128
pip install tensorrt==10.9.0.34 --force-reinstall

pip3 install --pre  torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128


pip install sentencepiece=="0.2.0" transformers=="4.48.2" accelerate=="1.3.0" diffusers=="0.32.2" protobuf=="5.29.3"

pip install notebook
pip install gradio safetensors peft pyinstrument
pip install nvidia-modelopt onnx torchprofile pulp onnxruntime
