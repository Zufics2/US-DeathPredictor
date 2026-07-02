import os
import json
import joblib

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from model import DeathPredictionModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device: {device}')

df = pd.read_csv("causes_deaths.csv")
os.makedirs("artifacts", exist_ok=True)

features = ["Year", "State", "Cause Name"]
target = "Age-adjusted Death Rate"

df = df[features + [target]].dropna()

df_encoded = pd.get_dummies(df, columns=["State", "Cause Name"], drop_first=True, dtype=np.float32)

X_df = df_encoded.drop(columns=[target])
y_df = df_encoded[target]

year_scaler = StandardScaler()
X_df['Year'] = year_scaler.fit_transform(X_df[['Year']])

X = X_df.values
y = y_df.values.astype(np.float32).reshape(-1, 1)

print(f'X: {X.shape}')
print(f'y: {y.shape}')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

input_size = X_train_tensor.shape[1]
model = DeathPredictionModel(input_size=input_size).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 250

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if (epoch + 1) % 25 == 0:
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch[{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}')

model.eval()
with torch.no_grad():
    test_predictions = model(X_test_tensor.to(device)).cpu().numpy()

mae = mean_absolute_error(y_test, test_predictions)
mse = mean_squared_error(y_test, test_predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, test_predictions)

print("\nMetrics")
print(f'MAE: {mae:.2f}')
print(f'MSE: {mse:.2f}')
print(f'RMSE: {rmse:.2f}')
print(f'R2 Score: {r2:.4f}')

metrics = {
    "mae": float(mae),
    "mse": float(mse),
    "rmse": float(rmse),
    "r2": float(r2)
}

print("\nSaving artifacts...")
torch.save(model.state_dict(), "artifacts/death_prediction_model.pth")
joblib.dump(year_scaler, "artifacts/x_scaler.joblib")
joblib.dump(X_df.columns.tolist(), "artifacts/feature_columns.pkl")

with open("artifacts/metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=4)

print("All artifacts saved successfully in 'artifacts/' directory!")