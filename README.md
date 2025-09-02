# Forecast Attention Model

A deep learning framework for time series forecasting using attention mechanisms. This repository provides an implementation of state-of-the-art neural architectures designed for accurate and interpretable time series prediction tasks such as energy consumption, stock price, weather data, and more.

## Features

- **Attention Mechanisms**: Leverage self-attention or transformer-based architectures to model temporal dependencies.
- **Flexible Data Pipeline**: Preprocess and feed various time series datasets with minimal configuration.
- **Custom Losses & Metrics**: Supports MSE, MAE, and custom metrics for robust evaluation.
- **Modular Design**: Easily extend or swap model components for experimentation.
- **Visualization Tools**: Plot attention weights, predictions, and error metrics for model interpretability.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Data Preparation](#data-preparation)
- [Model Architecture](#model-architecture)
- [Training & Evaluation](#training--evaluation)
- [Visualization](#visualization)
- [Customization](#customization)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/Ali-hey-0/forecast-Attention-Model.git
cd forecast-Attention-Model
pip install -r requirements.txt
```

**Main dependencies:**

- [PyTorch](https://pytorch.org/) for deep learning
- [NumPy](https://numpy.org/) for numerical computations
- [Pandas](https://pandas.pydata.org/) for data handling
- [Matplotlib](https://matplotlib.org/) and [Seaborn](https://seaborn.pydata.org/) for visualization

## Usage

**Basic workflow:**

1. Prepare your time series dataset (see [Data Preparation](#data-preparation)).
2. Configure model and training parameters in `config.yaml` or the script.
3. Train the model:
   ```bash
   python train.py --config config.yaml
   ```
4. Evaluate and visualize results:
   ```bash
   python evaluate.py --model-path runs/best_model.pth
   python visualize.py --model-path runs/best_model.pth
   ```

## Data Preparation

- Input data should be in CSV format with columns for features and target.
- Example format:
  ```
  timestamp,feature1,feature2,...,target
  2023-01-01 00:00,0.5,1.2,3.7
  ```
- Use `data_utils.py` for cleaning, normalization, and train/validation/test split.

## Model Architecture

The core model leverages the attention mechanism for sequence-to-sequence forecasting.

- **Encoder**: Embeds input sequence, applies positional encoding.
- **Attention Layer**: Computes attention scores to focus on relevant time steps.
- **Decoder**: Generates predictions for future steps.

Example model snippet (`model.py`):

```python
import torch.nn as nn

class ForecastAttentionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super().__init__()
        # Define encoder, attention, and decoder layers
    def forward(self, x):
        # Forward pass logic
        return out
```

## Training & Evaluation

Configure training parameters in `config.yaml`:

```yaml
epochs: 100
batch_size: 64
learning_rate: 0.001
optimizer: adam
loss: mse
```

Train the model:

```bash
python train.py --config config.yaml
```

Evaluate on test data:

```bash
python evaluate.py --model-path runs/best_model.pth
```

## Visualization

Visualize predictions and attention weights:

```bash
python visualize.py --model-path runs/best_model.pth
```

- Plots actual vs. predicted values
- Displays attention heatmaps for interpretability

## Customization

- **Model**: Modify `model.py` to change layers or attention mechanism.
- **Data**: Adjust `data_utils.py` for custom datasets.
- **Metrics**: Add or modify metrics in `metrics.py`.

## Results

| Dataset         | MAE   | RMSE  |
|-----------------|-------|-------|
| Energy (demo)   | 0.123 | 0.156 |
| Stock (demo)    | 0.091 | 0.114 |

*Note: Replace with your experiment results.*

## Contributing

Contributions are welcome! Please open issues or pull requests with improvements, bug fixes, or new features.

## License

This project is licensed under the [MIT License](LICENSE).

---

**Contact:** For questions, open an issue or reach out at [Ali-hey-0](https://github.com/Ali-hey-0).
