import torch
import matplotlib.pyplot as plt
import csv

file = open('length_weight.csv')

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

class LinearRegressionModel: 
    def __init__(self):
        self.W = torch.tensor([[0.0]], requires_grad=True)
        self.b = torch.tensor([[0.0]], requires_grad=True)
    
    def f(self, x):
        return x @ self.W + self.b
    
    def loss(self, x, y):
        return torch.mean(torch.square(self.f(x_tensor) - y_tensor))

model = LinearRegressionModel()

optimizer = torch.optim.SGD([model.W, model.b], 0.01)
for epoch in range(1000):
    model.loss(x_tensor, y_tensor).backward()
    optimizer.step()
    optimizer.zero_grad()

print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(x_tensor, y_tensor)))

plt.plot(x_tensor, y_tensor, 'o', label='$(x^{(i)},y^{(i)})$')
plt.xlabel('Lengde')
plt.ylabel('Vekt')
x = torch.tensor([[torch.min(x_tensor)], [torch.max(x_tensor)]])
y = model.f(x).detach()
plt.plot(x, y, color='orange', label='$\\hat y = f(x) = xW+b$')
plt.legend()
plt.show()