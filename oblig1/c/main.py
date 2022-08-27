import torch
import matplotlib.pyplot as plt
import csv

file = open('day_head_circumference.csv')

csvreader = csv.reader(file, delimiter=',')

header = []
header = next(csvreader)

x_observe = []
y_observe = []

for row in csvreader:
    x_observe.append(float(row[0]))
    y_observe.append(float(row[1]))

x_train = torch.tensor(x_observe).reshape(-1,1)
y_train = torch.tensor(y_observe).reshape(-1,1)

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

optimizer = torch.optim.SGD((model.b, model.W), 0.00000001)
for epoch in range(10000):
    model.loss(x_train, y_train).backward()
    optimizer.step()
    optimizer.zero_grad()

print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(x_train, y_train)))

plt.plot(x_train, y_train, 'o', label='$(x^{(i)},y^{(i)})$')
plt.xlabel('x')
plt.ylabel('y')
x = torch.arange(torch.min(x_train), torch.max(x_train), 1.0).reshape(-1,1)
y = model.f(x).detach()
plt.plot(x, y, color='orange', label='$f(x) = 20\sigma(xW + b) + 31$ \n$\sigma(z) = \dfrac{1}{1+e^{-z}}$')
plt.legend()
plt.show()