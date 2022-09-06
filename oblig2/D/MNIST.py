import torch
import torchvision
import matplotlib.pyplot as plt

import tqdm

# Load observations from the mnist dataset into training set
mnist_train = torchvision.datasets.MNIST('./data', train=True, download=True)
#print(mnist_train.data.shape)
x_train = mnist_train.data.reshape(-1, 784).float()  # Reshape input
#print(x_train.shape)
y_train = torch.zeros((mnist_train.targets.shape[0], 10))  # Create output tensor
#print(y_train.shape)
y_train[torch.arange(mnist_train.targets.shape[0]), mnist_train.targets] = 1  # Sets output for what the picture is supposed to be (1,2,3,4,...,9)


class HandWrittenNumbers:

    def __init__(self):
        self.W = torch.ones([784, 10], requires_grad=True)
        self.b = torch.ones([1, 10], requires_grad=True)

    # Predictor
    def f(self, x):
        return torch.nn.functional.softmax(self.logits(x), dim=1)

    def logits(self, x):
        return x @ self.W + self.b

    def accuracy(self, x, y):
        return torch.mean(torch.eq(self.f(x).argmax(1), y.argmax(1)).float())

    def loss(self, x, y):
        return torch.nn.functional.cross_entropy(self.logits(x), y.argmax(1))


model = HandWrittenNumbers()

# Optimizer
optimizer = torch.optim.SGD([model.W, model.b], 0.5)
for epoch in tqdm.tqdm(range(1000)):
    model.loss(x_train, y_train).backward()
    if epoch % 100 == 0:
        print("W = %s, b = %s, loss = %s" % (model.W.data, model.b.data, model.loss(x_train, y_train).data))
    optimizer.step()  
    optimizer.zero_grad()

print("W = %s, b = %s, loss = %s" % (model.W.data, model.b.data, model.loss(x_train, y_train).data))

# Test dataset
mnist_test = torchvision.datasets.MNIST('./data', train=False, download=True)
x_test = mnist_test.data.reshape(-1, 784).float()  # Reshape input
y_test = torch.zeros((mnist_test.targets.shape[0], 10))  # Create output tensor
y_test[torch.arange(mnist_test.targets.shape[0]), mnist_test.targets] = 1  # Populate output

print("\nAccuracy: " + str(model.accuracy(x_test, y_test)))

fig = plt.figure("MNIST-datasett")
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(model.W[:, i].detach().numpy().reshape(28, 28))
    plt.title(f'W: {i}')
    plt.xticks([])
    plt.yticks([])

plt.show()