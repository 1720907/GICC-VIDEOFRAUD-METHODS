# CUDA and cuDNN Installation Guide

## CUDA Installation  
1. Go to the official CUDA downloads page:  
   (https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=deb_local)

## cuDNN Installation  
1. Visit the cuDNN downloads page:  
   (https://developer.nvidia.com/cudnn-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=24.04&target_type=deb_local)

## Verify CUDA Installation

Add the following lines to your shell configuration file (`~/.bashrc`, `~/.zshrc`, etc.):

```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
source ~/.bashrc
```

It is necessary to reboot system.

## Check cuDNN Version

Run the following command in your terminal:

```bash
dpkg -l | grep libcudnn
```

This will list the installed cuDNN packages and their versions.


## Notes
- Ensure that your GPU drivers are correctly installed and recognized (`nvidia-smi`).
- Use `nvcc --version` or `cat /usr/local/cuda/version.txt` to check your CUDA version.