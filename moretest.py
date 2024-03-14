import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Load earthquake data from CSV file
data = pd.read_csv('sampledata.csv')


# Update column names to match the extraction code
data.columns = ['Date', 'Time', 'Latitude', 'Longitude', 'Type', 'Depth', 'Depth Error', 
                'Depth Seismic Stations', 'Magnitude', 'Magnitude Type', 'Magnitude Error', 
                'Magnitude Seismic Stations', 'Azimuthal Gap', 'Horizontal Distance', 
                'Horizontal Error', 'Root Mean Square', 'ID', 'Source', 'Location Source', 
                'Magnitude Source', 'Status']

# Extract features and labels
X = data[['Latitude', 'Longitude', 'Depth']].values.astype(np.float32)
y = data['Magnitude'].values.astype(np.float32)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train)
X_test_tensor = torch.tensor(X_test)
y_train_tensor = torch.tensor(y_train).view(-1, 1)
y_test_tensor = torch.tensor(y_test).view(-1, 1)

# Define the neural network architecture
class NeuralNet(nn.Module):
    def __init__(self, input_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Initialize the model
model = NeuralNet(input_size=X_train.shape[1])

# Move the model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

# Train the model
num_epochs = 50
batch_size = 32
for epoch in range(num_epochs):
    for i in range(0, len(X_train_tensor), batch_size):
        inputs = X_train_tensor[i:i+batch_size].to(device)
        targets = y_train_tensor[i:i+batch_size].to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model
with torch.no_grad():
    y_pred = model(X_test_tensor.to(device)).cpu().numpy()
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
