import torch
import numpy as np


data = [[1,2], [3,4]]
np_array = np.array(data)

# 데이터로 부터 직접 텐서 생성
x_data = torch.tensor(data)

# numpy 배열로부터 생성
x_np = torch.from_numpy(np_array)

print(x_data)
print(x_np)

# 명시적으로 재정의 하지 않는다면, 인자로 주어진 텐서의 속성(모양(2x3, 3x4 등 ), 자료형(int, float 등))을 유지
x_ones = torch.ones_like(x_data)
# 명시적으로 모양은 그대로 두고 자료형을 재정의함(float으로 변환)
x_rand = torch.rand_like(x_data, dtype=torch.float)

print(x_ones)
print(x_rand)

# shape는 텐서의 차원을 나타내는 튜플이다.
# 출력할 텐서의 차원을 결정한다.
shape = (2,3)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(rand_tensor)
print(ones_tensor)
print(zeros_tensor)