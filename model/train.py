import numpy as np
import torch as T
import requests
import tqdm

device = T.device('cpu')

class MNIST_Dataset(T.utils.data.Dataset):
    def __init__(self, images_data, labels_data):
        self.x_data, self.y_data = self.load_data(images_data, labels_data)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]

    def load_data(self, images_data, labels_data):
        images = self.load_idx_images(images_data)
        labels = self.load_idx_labels(labels_data)
        images = images.reshape(-1, 1, 28, 28)  # Reshape to [batch_size, channels, height, width]
        images = T.tensor(images, dtype=T.float32).to(device)
        labels = T.tensor(labels, dtype=T.int64).to(device)
        print(f"Loaded images shape {images.shape} and labels shape {labels.shape}")
        return images, labels

    def load_idx_images(self, data):
        images = np.array(data).reshape(len(data), 28 * 28)
        return images

    def load_idx_labels(self, data):
        labels = np.array(data)
        return labels

class Net(T.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = T.nn.Conv2d(1, 32, 5)
        self.conv2 = T.nn.Conv2d(32, 64, 5)
        self.fc1 = T.nn.Linear(1024, 512)
        self.fc2 = T.nn.Linear(512, 256)
        self.fc3 = T.nn.Linear(256, 10)
        self.pool1 = T.nn.MaxPool2d(2, stride=2)
        self.pool2 = T.nn.MaxPool2d(2, stride=2)
        self.drop1 = T.nn.Dropout(0.25)
        self.drop2 = T.nn.Dropout(0.50)
  
    def forward(self, x):
        z = T.relu(self.conv1(x))
        z = self.pool1(z)
        z = self.drop1(z)
        z = T.relu(self.conv2(z))
        z = self.pool2(z)
        z = z.view(-1, 1024)
        z = T.relu(self.fc1(z))
        z = self.drop2(z)
        z = T.relu(self.fc2(z))
        z = self.fc3(z)
        return z

def accuracy(model, ds):
    ldr = T.utils.data.DataLoader(ds, batch_size=len(ds), shuffle=False)
    n_correct = 0
    for data in ldr:
        (pixels, labels) = data
        with T.no_grad():
            oupts = model(pixels)
        _, predicteds = T.max(oupts, 1)
        n_correct += (predicteds == labels).sum().item()

    acc = (n_correct * 1.0) / len(ds)
    return acc

def main():
    # 0. Setup
    print("\nBegin MNIST with CNN demo")
    np.random.seed(1)
    T.manual_seed(1)

    # 1. Fetch MNIST Dataset from GoLang HTTP Server
    print("\nFetching MNIST training Dataset from GoLang HTTP Server")

    num_items = 20000
    url = f'http://localhost:8080/mnist_data?num={num_items}'
    response = requests.get(url)

    if response.status_code != 200:
        print(f"Error fetching data from server: {response.status_code}")
        return

    data = response.json()
    images_data = data['images']
    labels_data = data['labels']

    train_ds = MNIST_Dataset(images_data, labels_data)

    batch_size = 50
    train_loader = T.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # 2. Create Network
    print("\nCreating CNN network with 2 conv and 3 linear layers")
    net = Net().to(device)

    # 3. Train Model
    max_epochs = 20
    epoch_log_interval = 1
    learning_rate = 0.001
    
    loss_func = T.nn.CrossEntropyLoss()
    optimizer = T.optim.SGD(net.parameters(), lr=learning_rate)
    
    print(f"\nbatch_size = {batch_size}")
    print(f"loss = {loss_func}")
    print("optimizer = SGD")
    print(f"max_epochs = {max_epochs}")
    print(f"lrn_rate = {learning_rate}")

    print("\nStarting training")
    net.train()
    for epoch in range(max_epochs):
        epoch_loss = 0
        for (X, y) in tqdm.tqdm(train_loader):
            optimizer.zero_grad()
            outputs = net(X)
            loss = loss_func(outputs, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if epoch % epoch_log_interval == 0:
            print(f"Epoch {epoch + 1}/{max_epochs}, Loss: {epoch_loss:.4f}")
        if epoch % 5 == 0:
            acc = accuracy(net, train_ds)
            print(f"Accuracy on training set: {acc * 100:.2f}%")

    # 4. Evaluate Model
    print("\nComputing model accuracy on the training set")
    net.eval()
    acc = accuracy(net, train_ds)
    print(f"Accuracy on training set: {acc * 100:.2f}%")

    print("\nEnd MNIST demo")

if __name__ == "__main__":
    main()
