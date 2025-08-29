import torch

tensor_1 = torch.ones(4,4)

# GPU가 존재하면 텐서를 이동
if torch.cuda.is_available() :
    # 텐서를 cuda 장치로 보냄
    tensor = tensor_1.to("cuda")
    print(tensor.device)

# NumPy 식의 표준 인덱싱과 슬라이싱

# tensor[행,열]의 형식으로 원하는 위치를 지정한다.
# : (콜론) 은 모든 것을 의미한다.
# 16번 코드는 모든 행의 1번째 열을 모두 0으로 바꾼다는 뜻이다.
tensor[:,1] = 0
print(tensor)

# 텐서 합치기
# 주어진 차원에 따라 일련의 텐서를 연결할 수 있다.
# [] 안에는 이어 붙일 텐서를 나열하고
# dim = x 는 어떤 방향으로 텐서를 붙일 것인지를 의미한다.
# 원본 텐서의 차원에 따라 x에 들어갈 수 있는 수가 정해져있다 (2차원일 땐 0(y값)과 1(x값), 3차원일땐 0(z값), 1(y값), 2(x값))
# 숫자가 작은쪽에서 큰 쪽으로 커질수록 더 깊은 차원이 복사 된다고 생각하면 편하다.
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

# 텐서 곱하기
# 요소별 곱
data = [[1,2],[3,4]]
tensor_2 = torch.tensor(data)

# 두 문장은 표현만 다를뿐 완전히 같은 결과를 낸다!
# 두 텐서를 포개놓고 같은 위치에 있는 요소들끼리 곱한 값을 반환한다.
# 두 텐서가 모양이 다르면 오류 발생!
print(tensor_2.mul(tensor_2))
print(tensor_2 * tensor_2)

# 행렬 곱
data = [[1,2,3],[4,5,6]]
tensor_3 = torch.tensor(data)

# tensor.T 는 전치행렬(원본 텐서의 행과 열을 뒤바꾼 것)을 의미한다.
# 이해가 잘 안되면 손으로 직접 그려가면서 답을 도출해보자.
print(tensor_3.matmul(tensor_3.T))
print(tensor_3 @ tensor_3.T)