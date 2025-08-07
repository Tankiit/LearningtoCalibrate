# Learning2Reject (L2R) Framework

A unified framework for Learning to Reject (L2R) that combines:
- Safety rejection (harmful content)
- Domain shift rejection (out-of-distribution data)
- Conformal prediction (theoretical coverage guarantees)

## Features

- **Safety L2R**: Rejects harmful or unsafe content
- **Domain L2R**: Rejects out-of-distribution predictions
- **Conformal Prediction**: Provides theoretical coverage guarantees
- **ERM Baseline**: Comparison with standard empirical risk minimization
- **Unified Uncertainty**: Combines multiple sources of uncertainty
- **Domain Generalization**: Advanced domain adaptation with L2R

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Simple L2R Framework
```python
# Run the unified L2R demonstration
python simplel2r.py
```

### Domain Generalization with L2R
```bash
# Activate the torch-multimodal environment
source ../torch-multimodal/bin/activate

# Basic usage with default settings
python dg.py

# Custom training parameters
python dg.py --epochs 50 --batch_size 128 --lr 0.0001

# Save model and results
python dg.py --save_model --results_file my_results.json

# Load pre-trained model
python dg.py --load_model trained_model.pth

# Show help
python dg.py --help
```

### Command Line Arguments for dg.py

- `--data_dir`: Directory to store/load data (default: `/Users/mukher74/research/data`)
- `--num_domains`: Number of domains to generate (default: 4)
- `--num_classes`: Number of classes (default: 5)
- `--input_dim`: Input feature dimension (default: 20)
- `--hidden_dim`: Hidden layer dimension (default: 128)
- `--batch_size`: Batch size for training (default: 64)
- `--epochs`: Number of training epochs (default: 20)
- `--lr`: Learning rate (default: 0.001)
- `--save_model`: Save the trained model
- `--load_model`: Path to load a pre-trained model
- `--results_file`: File to save results (default: results.json)

## Project Structure

- `simplel2r.py`: Main implementation of the unified L2R framework
- `dg.py`: Domain generalization with L2R framework (with argparse support)
- `l2r.py`: Extended implementation with additional features
- `HarmBench/`: Benchmark suite for testing harmful content detection

## Data Directory

The framework uses `/Users/mukher74/research/data` as the default data directory for:
- Saving trained models
- Storing experimental results
- Loading pre-trained models

## License

MIT License 