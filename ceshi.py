import torch

# 检查 PyTorch 版本
print(f"PyTorch 版本: {torch.__version__}")

# 检查 CUDA 是否可用
if torch.cuda.is_available():
    print("CUDA 可用")
    print(f"GPU 数量: {torch.cuda.device_count()}")
    print(f"当前 GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA 版本: {torch.version.cuda}")

    # 测试 GPU 计算
    x = torch.tensor([1.0, 2.0]).cuda()
    y = torch.tensor([3.0, 4.0]).cuda()
    z = x + y
    print(f"GPU 计算结果: {z}")
else:
    print("CUDA 不可用，使用 CPU 计算")

    # 测试 CPU 计算
    x = torch.tensor([1.0, 2.0])
    y = torch.tensor([3.0, 4.0])
    z = x + y
    print(f"CPU 计算结果: {z}")


# 创建简单模型测试
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)

    def forward(self, x):
        return self.linear(x)


model = SimpleModel()
if torch.cuda.is_available():
    model.cuda()

# 测试前向传播
input_tensor = torch.randn(1, 2)
if torch.cuda.is_available():
    input_tensor = input_tensor.cuda()

output = model(input_tensor)
print(f"模型输出: {output}")