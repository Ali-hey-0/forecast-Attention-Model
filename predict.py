# predict.py (نسخه کامل با خروجی به CSV)
import torch
import matplotlib.pyplot as plt
from dataset import WeatherDataset
from model import Seq2Seq
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import mean_absolute_error
import pandas as pd
import os

# 💻 دستگاه
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(torch.get_num_threads())  # یا بذار روی عدد بالا مثل:
torch.set_num_threads(4)  # یا بیشتر بسته به تعداد واقعی core‌ها



# ⚙️ هایپرپارامتر
input_dim = 1
hidden_dim = 64
window_size = 168
output_window = 72
batch_size = 1

# 🧠 بارگذاری مدل
model = Seq2Seq(input_dim=input_dim, hidden_dim=hidden_dim, output_window=output_window).to(device)
model.load_state_dict(torch.load("./checkpoints/final_attention_model.pth", map_location=device))
model.eval()

# 📦 دیتاست تست
dataset = WeatherDataset("weather.csv", input_window=window_size, output_window=output_window)
test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# 📊 لیست برای ذخیره نتایج
all_preds = []
all_trues = []

# 🔍 پیش‌بینی روی کل test set
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        y_pred = model(x)
        
        all_preds.append(y_pred.squeeze(0).cpu().numpy())  # [72]
        all_trues.append(y.squeeze(0).cpu().numpy())       # [72]

# 📁 ساختار خروجی به DataFrame
df = pd.DataFrame({
    f"pred_{i+1}": [pred[i] for pred in all_preds]
    for i in range(output_window)
})
df_truth = pd.DataFrame({
    f"true_{i+1}": [true[i] for true in all_trues]
    for i in range(output_window)
})
results = pd.concat([df, df_truth], axis=1)

# 💾 ذخیره در فایل CSV
os.makedirs("results", exist_ok=True)
results.to_csv("results/predictions_vs_truth.csv", index=False)
print("✅ نتایج در فایل results/predictions_vs_truth.csv ذخیره شد.")
