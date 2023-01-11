import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric.nn as gnn
from torch_geometric.data import DataLoader
import seaborn as sns
import pandas as pd
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from sklearn.manifold import TSNE
import numpy as np
dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())

class GCN(nn.Module):
    def __init__(self, embedding_size=16):
        # Init parent
        super(GCN, self).__init__()
        torch.manual_seed(42)
        self.conv1 = gnn.GCNConv(dataset.num_features, embedding_size)
        self.drop_out = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self.conv2 = gnn.GCNConv(embedding_size, embedding_size)
        self.out = gnn.Linear(embedding_size, dataset.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.drop_out(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        x = self.drop_out(x)
        x = self. out(x)
        return x

def train(data, device, optimizer, model, loss_fn):
      optimizer.zero_grad()
      pred= model(data.x, data.edge_index)
      loss = loss_fn(pred[data.train_mask], data.y[data.train_mask])
      loss.backward()
      optimizer.step()
      return loss
def train_fn():
    model = GCN()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay= 5e-4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    data = dataset.data
    data = data.to(device)
    data_size = len(data)

    print("Starting training...")
    losses = []
    for epoch in range(2000):
        loss = train(data, device, optimizer, model, loss_fn)
        losses.append(loss)
        if epoch % 100 == 0:
            print(f"Epoch {epoch} | Train Loss {loss}")
    losses_float = [float(loss.cpu().detach().numpy()) for loss in losses]
    plt_ = sns.lineplot(losses_float)
    plt_.set(xlabel='epoch', ylabel='error')
    plt.savefig('train.png')
    print("Starting testing..")
    acc = test(model, data)
    print(f'the test accuracy is {acc}')
    out = model(data.x, data.edge_index)
    visualize(out, data.y)



def test(model, data):
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        test_correct = pred[data.test_mask] == data.y[data.test_mask]
        test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
        return test_acc
# def plt2arr(fig):
#     rgb_str = fig.canvas.tostring_rgb()
#     (w,h) = fig.canvas.get_width_height()
#     rgba_arr = np.fromstring(rgb_str, dtype=np.uint8, sep='').reshape((w,h,-1))
#     return rgba_arr


def visualize(out, color):
    fig = plt.figure(figsize=(5,5), frameon=False)
    z = TSNE(n_components=2).fit_transform(out.detach().cpu().numpy())
    plt.scatter(z[:, 0],
                z[:, 1],
                s=10,
                c=color.detach().cpu().numpy(),
                cmap="Set2"
                )
    plt.savefig('tsne.png')
    # fig.canvas.draw()


if __name__=='__main__':
    train_fn()