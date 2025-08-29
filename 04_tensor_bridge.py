import torch
import numpy as np

# cpu 상의 텐서와 NumPy 배열은 메모리 공간을 공유하기 때문에 하나를 변경하면 다른 하나도 변경된다.

# 텐서를 NumPy 배열로 변환하기
t = torch.ones(5)
print(f"t : {t}")
n = t.numpy()
print(f"n : {n}")

# 텐서의 변경 사항이 NumPy 배열에 반영된다
# 여기서 _ 연산은 원본 t 텐서 값을 직접 수정한다.
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

# NumPy 배열을 텐서로 변환하기
n = np.ones(5)
t = torch.from_numpy(n)

# NumPy 배열의 변경 사항이 텐서에 반영된다.
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")