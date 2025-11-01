import torch

# 1. 检查PyTorch版本
print("PyTorch版本:", torch.__version__)

# 2. 检查CUDA是否可用
print("CUDA是否可用:", torch.cuda.is_available())

# 3. 检查PyTorch编译时使用的CUDA版本
if torch.cuda.is_available():
    print("PyTorch对应的CUDA版本:", torch.version.cuda)
    # 4. 检查GPU设备数量及名称
    print("可用GPU数量:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA不可用。")