import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def function_to_predict(x):
    return x + np.sin(5*x)

X_MIN = -5
X_MAX = 5
NUM_DATA_POINTS = 100
NUM_LAYERS = 2
LAYER_SIZE = 100
EPOCHS = 3000

x_vals = torch.tensor(np.linspace(X_MIN,X_MAX, NUM_DATA_POINTS), dtype=torch.float32)
y_vals = function_to_predict(x_vals)


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.first_layer = nn.Linear(1, LAYER_SIZE)
        self.middle_layers = nn.ModuleList()
        for i in range(0, NUM_LAYERS):
            self.middle_layers.append(nn.Linear(LAYER_SIZE,LAYER_SIZE))
            self.middle_layers.append(nn.ReLU())
        self.last_layer = nn.Linear(LAYER_SIZE, 1)


    def forward(self, x):
        x = self.first_layer(x)
        for layer in self.middle_layers:
            x = layer(x)
        x = self.last_layer(x)
        return x

m = MyModel()
# prediction = m(x_vals[0].reshape(1,1))
# print(prediction)


loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(m.parameters(), lr=0.01)

BATCH_SIZE = 1000

for epoch in range(0, EPOCHS):
    for i in range(0, len(x_vals), BATCH_SIZE):
        m.train()
        batch_x = x_vals[i:i+BATCH_SIZE].reshape(-1,1)
        batch_y = y_vals[i:i+BATCH_SIZE].reshape(-1,1)
        predictions = m(batch_x)

        loss = loss_function(predictions, batch_y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print("loss: ", loss.item())

plot_x = x_vals
plot_y = y_vals
# Set model to evaluation mode and detach gradients for plotting
m.eval()
with torch.no_grad():
    plot_predictions = [m(torch.tensor(x, dtype=torch.float32).reshape(1,1)).detach().numpy().flatten()[0] for x in plot_x]

plt.plot(plot_x, plot_y, label="Actual", marker=".")
plt.plot(plot_x, plot_predictions, label="Predicted", marker=".")
plt.legend()
plt.title("Epoch: " + str(epoch))
plt.show()