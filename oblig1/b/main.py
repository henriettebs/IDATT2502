import torch
import matplotlib.pyplot as plt
import csv

file = open('day_length_weight.csv')

csvreader = csv.reader(file, delimiter=',')

header = []
header = next(csvreader)

x_train = []
y_train = []
z_train = []

for row in csvreader:
    x_train.append(float(row[0]))
    y_train.append(float(row[1]))
    z_train.append(float(row[2]))

x_train_tensor = torch.tensor(x_train).reshape(-1,1)
y_train_tensor = torch.tensor(y_train).reshape(-1,1)
z_train_tensor = torch.tensor(z_train).reshape(-1,1)

class LinearRegressionModel:
    def __init__(self):
        self.W1 = torch.tensor([[0.0]], requires_grad=True)
        self.W2 = torch.tensor([[0.0]], requires_grad=True)
        self.b = torch.tensor([[0.0]], requires_grad=True)

    def f(self, x1, x2):
        return x1 @ self.W1 + x2 @ self.W2 + self.b

    def loss(self, x1, x2, y):
        return torch.nn.functional.mse_loss(self.f(x1, x2), y)
    
model = LinearRegressionModel()

optimizer = torch.optim.SGD([model.W1, model.W2, model.b], 0.0001)
for epoch in range(100000):
    model.loss(x_train_tensor, y_train_tensor, z_train_tensor).backward()
    optimizer.step()
    optimizer.zero_grad()



print("W1 = %s, W2 = %s b = %s, loss = %s" % (model.W1, model.W2, model.b, model.loss(x_train_tensor, y_train_tensor, z_train_tensor)))

fig = plt.figure('Linear regression 3d')
ax = plt.axes(projection='3d', title="Predict days based on length and weight")
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$y$')
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.w_xaxis.line.set_lw(0)
ax.w_yaxis.line.set_lw(0)
ax.w_zaxis.line.set_lw(0)
ax.quiver([0], [0], [0], [torch.max(x_train_tensor + 1)], [0],
[0], arrow_length_ratio=0.05, color='black')
ax.quiver([0], [0], [0], [0], [torch.max(y_train_tensor + 1)],
[0], arrow_length_ratio=0.05, color='black')
ax.quiver([0], [0], [0], [0], [0], [torch.max(z_train_tensor + 1)],
arrow_length_ratio=0, color='black')

ax.scatter(x_train, y_train, z_train)
x = torch.tensor([[torch.min(x_train_tensor)], [torch.max(x_train_tensor)]])
y = torch.tensor([[torch.min(y_train_tensor)], [torch.max(y_train_tensor)]])
ax.plot(x.flatten(), y.flatten(), model.f(x, y).detach().flatten(), label='$f(x)=x1W1+x2W2+b$', color="orange")
ax.legend()
plt.show()


# plt.plot(x_train_tensor, y_train_tensor, z_train_tensor, 'o', label='$(x^{(i)},y^{(i)},z^{(i)})$')
# plt.xlabel('Lengde')
# plt.ylabel('Vekt')
# plt.show()
