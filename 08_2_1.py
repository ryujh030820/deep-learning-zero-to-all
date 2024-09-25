import torch

# 사용할 디바이스 설정 (GPU가 있으면 GPU를, 없으면 CPU를 사용)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 재현성을 위해 랜덤 시드 설정
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

# XOR 입력과 출력 정의
X = torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]]).to(device)
Y = torch.FloatTensor([[0], [1], [1], [0]]).to(device)

# 가중치와 편향 초기화
w1 = torch.randn(2, 2).to(device)
b1 = torch.zeros(2).to(device)
w2 = torch.randn(2, 1).to(device)
b2 = torch.zeros(1).to(device)

# 학습률 설정
learning_rate = 1

# 시그모이드 함수 정의
def sigmoid(x):
    return 1.0 / (1.0 + torch.exp(-x))

# 시그모이드 함수의 미분 정의
def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))

# 학습 과정
for step in range(10001):
    # 순전파
    l1 = torch.add(torch.matmul(X, w1), b1)  # 첫 번째 레이어의 선형 변환
    a1 = sigmoid(l1)                         # 활성화 함수 적용
    l2 = torch.add(torch.matmul(a1, w2), b2) # 두 번째 레이어의 선형 변환
    Y_pred = sigmoid(l2)                     # 출력값 계산

    # 코스트 계산
    cost = -torch.mean(Y * torch.log(Y_pred) + (1 - Y) * torch.log(1 - Y_pred))

    # 역전파 (체인 룰 사용)
    d_Y_pred = (Y_pred - Y) / (Y_pred * (1.0 - Y_pred) + 1e-7)  # 0으로 나누는 것을 방지 # cost를 Y_pred로 미분

    # 두 번째 레이어의 기울기 계산
    d_l2 = d_Y_pred * sigmoid_prime(l2) # sigmoid를 l2로
    d_b2 = d_l2
    d_w2 = torch.matmul(torch.transpose(a1, 0, 1), d_b2)

    # 첫 번째 레이어의 기울기 계산
    d_a1 = torch.matmul(d_b2, torch.transpose(w2, 0, 1))
    d_l1 = d_a1 * sigmoid_prime(l1)
    d_b1 = d_l1
    d_w1 = torch.matmul(torch.transpose(X, 0, 1), d_b1)

    # 가중치와 편향 업데이트
    w1 = w1 - learning_rate * d_w1
    b1 = b1 - learning_rate * torch.mean(d_b1, 0)
    w2 = w2 - learning_rate * d_w2
    b2 = b2 - learning_rate * torch.mean(d_b2, 0)

    # 100번마다 코스트 출력
    if step % 100 == 0:
        print(step, cost.item())

# 학습이 끝난 후 결과 출력
with torch.no_grad():
    l1 = torch.add(torch.matmul(X, w1), b1)
    a1 = sigmoid(l1)
    l2 = torch.add(torch.matmul(a1, w2), b2)
    Y_pred = sigmoid(l2)
    predicted = (Y_pred > 0.5).float()
    accuracy = (predicted == Y).float().mean()
    print('Prediction:', predicted.cpu().numpy())
    print('Accuracy:', accuracy.item())
