import torch as torch
import matplotlib.pyplot as plt
import numpy as np
import random

# Training input and output
x_train = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]).reshape(-1, 2)
y_train = torch.tensor([[0.0], [1.0], [1.0], [0.0]]).reshape(-1, 1)

class XOR:
    def __init__(self):
        self.W1 = torch.tensor([[random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)],
                                [random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)]],
                               requires_grad=True)
        self.b1 = torch.tensor([[random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)]],
                               requires_grad=True)
        self.W2 = torch.tensor([[random.uniform(-1.0, 1.0)], [random.uniform(-1.0, 1.0)]],
                               requires_grad=True)
        self.b2 = torch.tensor([[random.uniform(-1.0, 1.0)]],
                               requires_grad=True)

    # First layer
    def f1(self, x):
        return torch.sigmoid(x @ self.W1 + self.b1)

    # Second layer
    def f2(self, h):
        return torch.sigmoid(h @ self.W2 + self.b2)

    # Predictor
    def f(self, x):
        return self.f2(self.f1(x))

    # Cross Entropy
    def loss(self, x, y):
        return torch.nn.functional.binary_cross_entropy(self.f(x), y)


model = XOR()

optimizer = torch.optim.SGD([model.b1, model.b2, model.W1, model.W2], 0.1)
for epoch in range(200_000):
    loss = model.loss(x_train, y_train)
    loss.backward() 
    optimizer.step()
    optimizer.zero_grad()
    if epoch%1000==0:
        print("Epoch: {}, loss: {}".format(epoch, loss))

print(f'W1 = {model.W1}, W2 = {model.W2}, b1 = {model.b1}, b2 = {model.b2}, loss = {model.loss(x_train.reshape(-1, 2), y_train)}')


# Plot
fig = plt.figure('XOR-operatoren')
ax = fig.gca(projection='3d')
plt.title('XOR-operator')
# set axes limits, labels and create a table of the XOR
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(0, 1)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$y$')
plt.table(cellText=[[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]],
          colWidths=[0.1] * 3,
          colLabels=["$x_1$", "$x_2$", "$f(x)$"],
          cellLoc="center",
          loc="lower right")

x1 = np.arange(0, 1, 0.02)  
x2 = np.arange(0, 1, 0.02) 
# Calculate y-axis values
y = np.empty([len(x1), len(x2)], dtype=np.double)
for t in range(len(x1)):
    for r in range(len(x2)):
        y[t, r] = float(model.f(torch.tensor([float(x1[t]), float(x2[r])])))

x1, x2 = np.meshgrid(x1, x2)
surf = ax.plot_wireframe(x1, x2, np.array(y)) 

# Plotting points for f(x1, x2) in x_train
xer = [float(x[0]) for x in x_train]
yer = [float(x[1]) for x in x_train]
ax.scatter(xer, yer, y_train)

float(model.f(torch.tensor([1.0, 0.0])))

# Customize the view angle to see the scatter points lie on plane y=0
ax.view_init(elev=10, azim=-170)

plt.show()