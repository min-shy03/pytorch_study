import torch

tensor = torch.rand(3,4)

# 텐서의 속성 표시 함수들
print(tensor.shape)     # 텐서의 모양 (a X b)
print(tensor.dtype)     # 텐서의 데이터 타입(자료형)
print(tensor.device)    # 텐서가 저장되는 장치