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

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```python
# Run the unified L2R demonstration
python simplel2r.py
```

## Project Structure

- `simplel2r.py`: Main implementation of the unified L2R framework
- `l2r.py`: Extended implementation with additional features
- `HarmBench/`: Benchmark suite for testing harmful content detection

## License

MIT License 