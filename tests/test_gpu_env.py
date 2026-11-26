import pytest


@pytest.mark.gpu
def test_gpu_stack_imports():
    import gsplat  # noqa: F401
    import sam2  # noqa: F401
    import torch

    assert torch.cuda.is_available()
    props = torch.cuda.get_device_properties(0)
    print(f"\nGPU: {props.name}, {props.total_memory / 2**30:.1f} GB, "
          f"torch {torch.__version__}, CUDA {torch.version.cuda}, gsplat {gsplat.__version__}")
