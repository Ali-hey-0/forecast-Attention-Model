# train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import WeatherDataset
from model import Seq2Seq
import matplotlib.pyplot as plt
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ پارامترها
window_size = 168   # 1 هفته گذشته
horizon = 72        # 3 روز آینده
batch_size = 128
num_epochs = 5
hidden_dim = 64
lr = 0.001

# 📊 دیتاست
dataset = WeatherDataset("weather.csv", input_window=window_size, output_window=horizon)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 🧠 مدل
model = Seq2Seq(input_dim=1, hidden_dim=hidden_dim, output_window=horizon).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# 📁 ذخیره
os.makedirs("checkpoints", exist_ok=True)

losses = []
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    epoch_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
    losses.append(epoch_loss)

    # 💾 ذخیره مدل هر 5 epoch
    if (epoch + 1) % 5 == 0:
        torch.save(model.state_dict(), f"checkpoints/model_epoch{epoch+1}.pth")

# 📈 نمودار Loss
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.grid(True)
plt.show()

# 🧠 ذخیره نهایی
torch.save(model.state_dict(), "checkpoints/final_attention_model.pth")
