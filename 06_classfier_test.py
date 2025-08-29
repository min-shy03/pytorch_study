import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 이 스크립트가 직접 실행될 때만 아래 코드를 실행하도록 보호
if __name__ == '__main__':
    # 다운로드한 이미지를 신경망에 넣기전에 어떻게 처리할지 "가공할 레시피" 를 만드는 코드
    transform = transforms.Compose(
        # 불러온 이미지를 텐서 형태로 변환
        [transforms.ToTensor(),
        # 텐서의 값 범위를 -1.0 에서 1.0 사이로 정규화
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # 데이터 묶음 크기 (데이터를 한 번에 4개씩 처리하겠다는 의미)
    batch_size = 4

    # CIFAR 데이터셋의 학습용 데이터를 불러오는 코드
    # train이 False면 테스트용 데이터 가져옴
    # download가 True면 루트에 아무 데이터도 없을 경우 인터넷에서 자동으로 데이터 다운로드
    # transform = 데이터를 어떻게 가공할 것인지 (위에서 설정함)
    # 내 데이터를 적용 시킬때는 다른 방법 필요(노션에 정리 해놓음)
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)

    # 위에서 설정한 trainset 데이터셋을 효율적으로 신경망에 공급하는 코드
    # shuffle = 학습을 시작할 때마다 데이터 순서를 섞어 과적합 방지 (학습 효과 올라감)
    # num_worker = 데이터를 학습시키는 동안 나머지 워커가 다음 배치 (데이터)를 준비함 (학습 속도 향상)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    # CIFAR 데이터셋의 테스트용 데이터를 불러오는 코드
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)

    # 테스트용 데이터 공급기계
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # 이미지를 보여주기 위한 함수
    def imshow(img):
        img = img / 2 + 0.5     # 정규화 해제
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    # ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ

    # 시험용 데이터로 신경망 검사하기
    dataiter = iter(testloader)
    images, labels = next(dataiter)

    # 이미지 출력
    imshow(torchvision.utils.make_grid(images))
    print("GroundTruth : ", " ".join(f"{classes[labels[j]]:5s}" for j in range(4)))

    # 신경망 클래스
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = torch.flatten(x, 1) # 배치를 제외한 모든 차원을 평탄화(flatten)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    
    net = Net()
    PATH = "./cifar_net.pth"
    net.load_state_dict(torch.load(PATH))

    outputs = net(images)

    _, predicted = torch.max(outputs, 1)

    print("predicted: ", " ".join(f"{classes[predicted[j]]:5s}" for j in range(4)))

    correct = 0
    total = 0

    with torch.no_grad() :
        for data in testloader :
            images, labels = data
            # 신경망에 이미지를 통과시켜 출력을 계산
            outputs = net(images)
            # 가장 높은 값(energy)를 갖는 분류(class)를 정답으로 선택
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f"10000장 테스트 정확도 : {100 * correct // total}%")

    # 각 분류(class)에 대한 예측값 계산을 위해 준비
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # 변화도는 여전히 필요하지 않습니다
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            # 각 분류별로 올바른 예측 수를 모읍니다
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1


    # 각 분류별 정확도(accuracy)를 출력합니다
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')