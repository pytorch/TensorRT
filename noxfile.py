import nox
import os

# Use system installed Python packages
PYT_PATH='/opt/conda/lib/python3.8/site-packages' if not 'PYT_PATH' in os.environ else os.environ["PYT_PATH"]

# Root directory for torch_tensorrt. Set according to docker container by default
TOP_DIR='/torchtrt' if not 'TOP_DIR' in os.environ else os.environ["TOP_DIR"]

# Download the dataset
@nox.session(python=["3"], reuse_venv=True)
def download_datasets(session):
    print("Downloading dataset to path", os.path.join(TOP_DIR, 'examples/int8/training/vgg16'))
    session.chdir(os.path.join(TOP_DIR, 'examples/int8/training/vgg16'))
    session.run_always('wget', 'https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz', external=True)
    session.run_always('tar', '-xvzf', 'cifar-10-binary.tar.gz', external=True)
    session.run_always('mkdir', '-p', 
                        os.path.join(TOP_DIR, 'tests/accuracy/datasets/data'),
                        external=True)
    session.run_always('cp', '-rpf', 
                        os.path.join(TOP_DIR, 'examples/int8/training/vgg16/cifar-10-batches-bin'),
                        os.path.join(TOP_DIR, 'tests/accuracy/datasets/data/cidar-10-batches-bin'),
                        external=True)

# Download the model
@nox.session(python=["3"], reuse_venv=True)
def download_models(session):
    session.install('timm')
    session.chdir('tests/modules')
    session.run_always('python', 
                        'hub.py',
                        env={'PYTHONPATH': PYT_PATH})

# Train the model
@nox.session(python=["3"], reuse_venv=True)
def train_model(session):
    session.chdir(os.path.join(TOP_DIR, 'examples/int8/training/vgg16'))
    session.run_always('python', 
                        'main.py', 
                        '--lr', '0.01', 
                        '--batch-size', '128', 
                        '--drop-ratio', '0.15', 
                        '--ckpt-dir', 'vgg16_ckpts', 
                        '--epochs', '25',
                        env={'PYTHONPATH': PYT_PATH})

    # Export model
    session.run_always('python',
                        'export_ckpt.py',
                        'vgg16_ckpts/ckpt_epoch25.pth',
                        env={'PYTHONPATH': PYT_PATH})

# Finetune the model
@nox.session(python=["3"], reuse_venv=True)
def finetune_model(session):
    # Install pytorch-quantization dependency
    session.install('pytorch-quantization', '--extra-index-url', 'https://pypi.ngc.nvidia.com')

    session.chdir(os.path.join(TOP_DIR, 'examples/int8/training/vgg16'))
    session.run_always('python', 
                        'finetune_qat.py', 
                        '--lr', '0.01', 
                        '--batch-size', '128', 
                        '--drop-ratio', '0.15', 
                        '--ckpt-dir', 'vgg16_ckpts', 
                        '--start-from', '25', 
                        '--epochs', '26',
                        env={'PYTHONPATH': PYT_PATH})
    
    # Export model 
    session.run_always('python',
                        'export_qat.py',
                        'vgg16_ckpts/ckpt_epoch26.pth',
                        env={'PYTHONPATH': PYT_PATH})

# Run PTQ tests
@nox.session(python=["3"], reuse_venv=True)
def ptq_test(session):
    session.chdir(os.path.join(TOP_DIR, 'tests/py'))
    session.run_always('cp', '-rf', 
                        os.path.join(TOP_DIR, 'examples/int8/training/vgg16', 'trained_vgg16.jit.pt'), 
                        '.',
                        external=True)
    tests = [
        'test_ptq_dataloader_calibrator.py',
        'test_ptq_to_backend.py',
        'test_ptq_trt_calibrator.py'
        ]
    for test in tests:
        session.run_always('python', test,
                        env={'PYTHONPATH': PYT_PATH})

# Run QAT tests
@nox.session(python=["3"], reuse_venv=True)
def qat_test(session):
    session.chdir(os.path.join(TOP_DIR, 'tests/py'))
    session.run_always('cp', '-rf', 
                        os.path.join(TOP_DIR, 'examples/int8/training/vgg16', 'trained_vgg16_qat.jit.pt'), 
                        '.',
                        external=True)

    session.run_always('python',
                        'test_qat_trt_accuracy.py',
                        env={'PYTHONPATH': PYT_PATH})

# Run Python API tests
@nox.session(python=["3"], reuse_venv=True)
def api_test(session):
    session.chdir(os.path.join(TOP_DIR, 'tests/py'))
    tests = [
            "test_api.py",
            "test_to_backend_api.py"
            ]
    for test in tests:
        session.run_always('python',
                            test,
                            env={'PYTHONPATH': PYT_PATH})

# Clean up
@nox.session(reuse_venv=True)
def cleanup(session):
    target = [
        'examples/int8/training/vgg16/*.jit.pt',
        'examples/int8/training/vgg16/vgg16_ckpts',
        'examples/int8/training/vgg16/cifar-10-*',
        'examples/int8/training/vgg16/data',
        'tests/modules/*.jit.pt',
        'tests/py/*.jit.pt'
    ]

    target = ' '.join(x for x in [os.path.join(TOP_DIR, i) for i in target])
    session.run_always('bash', '-c',
                        str('rm -rf ') + target,
                        external=True)