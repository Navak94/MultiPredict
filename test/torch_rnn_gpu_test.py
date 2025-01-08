import torch
print("hi!")
print("CUDA Available:", torch.cuda.is_available())
print("CUDA Version:", torch.version.cuda)
print("GPU in Use:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU Detected")
