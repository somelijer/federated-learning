import ast
import json
import os
import time
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
        self.conv1 = T.nn.Conv2d(1, 10, kernel_size=5, stride=1)
        self.conv2 = T.nn.Conv2d(10, 10, kernel_size=5, stride=1)
        self.pool = T.nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = T.nn.Linear(4 * 4 * 10, 100)
        self.fc2 = T.nn.Linear(100, 10)
  
    def forward(self, x):
        x = T.relu(self.conv1(x))  # 24x24x10
        x = self.pool(x)  # 12x12x10
        x = T.relu(self.conv2(x))  # 8x8x10
        x = self.pool(x)  # 4x4x10
        x = x.view(-1, 4 * 4 * 10)  # flattening
        x = T.relu(self.fc1(x))
        x = self.fc2(x)
        return x

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

def request_initial_weights():
    url = 'http://localhost:8080/initial_weights'
    response = requests.get(url)
    if response.status_code == 200:
        weights = response.json()
        return weights
    elif response.status_code == 404:
        print("No initial weights available.")
        return None
    else:
        print(f"Error fetching initial weights: {response.status_code}, {response.text}")
        return None

def load_weights_to_model(model, weights):
    with open('out.txt', 'w') as f:
        f.write('conv1_weight shape:\n')
        f.write(f"{T.tensor(weights['conv1_weight'], dtype=T.float32).shape}\n")
        f.write('conv1_bias shape:\n')
        f.write(f"{T.tensor(weights['conv1_bias'], dtype=T.float32).shape}\n")
        f.write('conv2_weight shape:\n')
        f.write(f"{T.tensor(weights['conv2_weight'], dtype=T.float32).shape}\n")
        f.write('conv2_bias shape:\n')
        f.write(f"{T.tensor(weights['conv2_bias'], dtype=T.float32).shape}\n")
        f.write('fc1_weight shape:\n')
        f.write(f"{T.tensor(weights['fc1_weight'], dtype=T.float32).shape}\n")
        f.write('fc1_bias shape:\n')
        f.write(f"{T.tensor(weights['fc1_bias'], dtype=T.float32).shape}\n")
        f.write('fc2_weight shape:\n')
        f.write(f"{T.tensor(weights['fc2_weight'], dtype=T.float32).shape}\n")
        f.write('fc2_bias shape:\n')
        f.write(f"{T.tensor(weights['fc2_bias'], dtype=T.float32).shape}\n")

    model.conv1.weight.data = T.tensor(weights['conv1_weight'], dtype=T.float32).to(device)
    model.conv1.bias.data = T.tensor(weights['conv1_bias'], dtype=T.float32).to(device)
    model.conv2.weight.data = T.tensor(weights['conv2_weight'], dtype=T.float32).to(device)
    model.conv2.bias.data = T.tensor(weights['conv2_bias'], dtype=T.float32).to(device)
    model.fc1.weight.data = T.tensor(weights['fc1_weight'], dtype=T.float32).to(device)
    model.fc1.bias.data = T.tensor(weights['fc1_bias'], dtype=T.float32).to(device)
    model.fc2.weight.data = T.tensor(weights['fc2_weight'], dtype=T.float32).to(device)
    model.fc2.bias.data = T.tensor(weights['fc2_bias'], dtype=T.float32).to(device)

def send_weights_to_golang(weights):
    url = 'http://localhost:8080/weights'  # Assuming the Go server listens on port 8081
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, data=json.dumps(weights), headers=headers)
    if response.status_code == 200:
        print("Weights sent successfully")
    else:
        print(f"Error sending weights: {response.status_code}, {response.text}")

def convert_weights_to_dict(model):
    weights = {
        'conv1_weight': model.conv1.weight.detach().cpu().numpy().tolist(),
        'conv1_bias': model.conv1.bias.detach().cpu().numpy().tolist(),
        'conv2_weight': model.conv2.weight.detach().cpu().numpy().tolist(),
        'conv2_bias': model.conv2.bias.detach().cpu().numpy().tolist(),
        'fc1_weight': model.fc1.weight.detach().cpu().numpy().tolist(),
        'fc1_bias': model.fc1.bias.detach().cpu().numpy().tolist(),
        'fc2_weight': model.fc2.weight.detach().cpu().numpy().tolist(),
        'fc2_bias': model.fc2.bias.detach().cpu().numpy().tolist()
    }
    return weights

def save_model_to_text(model, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        for name, param in model.named_parameters():
            f.write(f"{name}\n")
            f.write(f"{param.detach().cpu().numpy().tolist()}\n")

def load_model_from_text(model, filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        param_dict = {}
        for i in range(0, len(lines), 2):
            name = lines[i].strip()
            param = ast.literal_eval(lines[i+1].strip())
            param_dict[name] = T.tensor(param, dtype=T.float32).to(device)

        model.load_state_dict(param_dict)

def main():
    np.random.seed(1)
    T.manual_seed(1)

    # 1. Fetch MNIST Dataset from GoLang HTTP Server
    print("\nFetching MNIST training Dataset from GoLang HTTP Server")

    num_items = 10  # DEPRECATED
    url = f'http://localhost:8080/mnist_data?num={num_items}'

    # Ping the server every second for 15 seconds
    max_retries = 15
    for _ in range(max_retries):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                break
        except requests.exceptions.ConnectionError:
            print("Server is not up yet, retrying...")
        time.sleep(1)
    else:
        print("Server did not respond within 15 seconds. Exiting.")
        return

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
    print("\nStarting training script")
    net = Net().to(device)

    initial_weights = request_initial_weights()
    if initial_weights:
        load_weights_to_model(net, initial_weights)
        print("\nLoaded in existing weights")
    else:
        print("\nInitialising random weights")

    # 3. Train Model
    max_epochs = 2
    epoch_log_interval = 1
    learning_rate = 0.001
    
    loss_func = T.nn.CrossEntropyLoss()
    optimizer = T.optim.Adam(net.parameters(), lr=learning_rate)
    
    print(f"\nbatch_size = {batch_size}")
    print(f"loss = {loss_func}")
    print("optimizer = Adam")
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
        if (epoch+1) % 5 == 0:
            acc = accuracy(net, train_ds)
            print(f"Accuracy on training set: {acc * 100:.2f}%")

    net.eval()
    if True:
        acc = accuracy(net, train_ds)
        print(f"Accuracy on training set: {acc * 100:.2f}%")

    send_weights_to_golang(convert_weights_to_dict(net))
    save_model_to_text(net, "neural_model/mnist_model.txt")

    print("\nEnd training script")

if __name__ == "__main__":
    main()
