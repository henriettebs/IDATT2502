import torch
import matplotlib.pyplot as plt
import csv

file = open('day_head_circumference.csv')

filereader = csv.reader(file, delimiter=',')

header = []
header = next(filereader)

x_train = []
y_train = []

for row in filereader:
    x_train.append(float(row[0]))
    y_train.append(float(row[1]))

x_tensor = torch.tensor(x_train).reshape(-1,1)
y_tensor = torch.tensor(y_train).reshape(-1,1)

class RegressionModel:
    def __init__(self):
        self.W = torch.tensor([[0.0]], requires_grad=True)
        self.b = torch.tensor([[0.0]], requires_grad=True)

    def f(self, x):
        return 20 * self.sigmoid(x @ self.W + self.b) + 31

    def loss(self, x, y):
        return torch.mean(torch.square(self.f(x) - y))
    
    def sigmoid(self, z):
        return 1 / (1 +torch.exp(-z))

model = RegressionModel()

optimizer = torch.optim.SGD((model.b, model.W), 0.00000001) #Adam er en raskere optim funksjon
for epoch in range(10000):
    model.loss(x_tensor, y_tensor).backward()
    optimizer.step()
    optimizer.zero_grad()

print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(x_tensor, y_tensor)))

plt.plot(x_tensor, y_tensor, 'o', label='$(x^{(i)},y^{(i)})$')
plt.xlabel('Dag')
plt.ylabel('Hodeomkrets')
x = torch.arange(torch.min(x_tensor), torch.max(x_tensor), 1.0).reshape(-1,1)
y = model.f(x).detach()
plt.plot(x, y, color='orange', label='$f(x) = 20\sigma(xW + b) + 31$ \n$\sigma(z) = \dfrac{1}{1+e^{-z}}$')
plt.legend()
plt.show()