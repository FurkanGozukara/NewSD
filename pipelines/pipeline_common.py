import torch
import gc
from torch import nn
from accelerate.hooks import AlignDevicesHook, CpuOffload, remove_hook_from_module

def torch_gc():

    if torch.cuda.is_available():
        with torch.cuda.device('cuda'):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
   
    gc.collect()
def optionally_disable_offloading(_pipeline):

    """
    Optionally removes offloading in case the pipeline has been already sequentially offloaded to CPU.

    Args:
        _pipeline (`DiffusionPipeline`):
            The pipeline to disable offloading for.

    Returns:
        tuple:
            A tuple indicating if `is_model_cpu_offload` or `is_sequential_cpu_offload` is True.
    """
    is_model_cpu_offload = False
    is_sequential_cpu_offload = False
    print(
            fr"Restarting CPU Offloading for {_pipeline.unet_name}..."
          )
    if _pipeline is not None:
        for _, component in _pipeline.components.items():
            if isinstance(component, nn.Module) and hasattr(component, "_hf_hook"):
                if not is_model_cpu_offload:
                    is_model_cpu_offload = isinstance(component._hf_hook, CpuOffload)
                if not is_sequential_cpu_offload:
                    is_sequential_cpu_offload = isinstance(component._hf_hook, AlignDevicesHook)

               
                remove_hook_from_module(component, recurse=True)

    return (is_model_cpu_offload, is_sequential_cpu_offload)