import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Load earthquake data from CSV file
data = pd.read_csv('database.csv')

# Extract features and labels
X = data[['Latitude', 'Longitude', 'Depth']].values.astype(np.float32)
y = data['Magnitude'].values.astype(np.float32)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train).view(-1, 1))
test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test).view(-1, 1))

# Define data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Define the neural network architecture
class NeuralNet(nn.Module):
    def __init__(self, input_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.2)
        self.batch_norm1 = nn.BatchNorm1d(256)
        self.batch_norm2 = nn.BatchNorm1d(512)
        self.batch_norm3 = nn.BatchNorm1d(256)
        self.batch_norm4 = nn.BatchNorm1d(128)
        self.batch_norm5 = nn.BatchNorm1d(64)

    def forward(self, x):
        x = self.batch_norm1(torch.relu(self.fc1(x)))
        x = self.dropout(x)
        x = self.batch_norm2(torch.relu(self.fc2(x)))
        x = self.dropout(x)
        x = self.batch_norm3(torch.relu(self.fc3(x)))
        x = self.dropout(x)
        x = self.batch_norm4(torch.relu(self.fc4(x)))
        x = self.dropout(x)
        x = self.batch_norm5(torch.relu(self.fc5(x)))
        x = self.fc6(x)
        return x

# Function to calculate evaluation metrics
def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return mse, mae, rmse

# Function for model training and evaluation
def train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs=50):
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        train_loss = 0.0
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        # Calculate validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

        # Average loss over dataset
        train_loss /= len(train_loader.dataset)
        val_loss /= len(test_loader.dataset)

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss}, Validation Loss: {val_loss}')

        # Update learning rate scheduler
        scheduler.step(val_loss)

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')

    # Load the best model
    best_model = NeuralNet(input_size=X_train.shape[1])
    best_model.load_state_dict(torch.load('best_model.pth'))
    best_model.eval()

    # Evaluate the best model
    test_loss = 0.0
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = best_model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item() * inputs.size(0)
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(outputs.cpu().numpy())

    test_loss /= len(test_loader.dataset)
    mse, mae, rmse = calculate_metrics(np.array(y_true), np.array(y_pred))
    print("Final Test Loss:", test_loss)
    print("Mean Squared Error:", mse)
    print("Mean Absolute Error:", mae)
    print("Root Mean Squared Error:", rmse)

# Initialize the model
model = NeuralNet(input_size=X_train.shape[1])

# Move the model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Learning rate scheduler
scheduler = ReduceLROnPlateau(optimizer, 'min')

# Train and evaluate the model
train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, scheduler)
