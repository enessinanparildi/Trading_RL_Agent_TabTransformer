# Trading_RL_Agent_TabTransformer
A PPO based stock trading RL agent implementation using TabTransformer 

This project implements a Reinforcement Learning agent for stock trading using the Stable Baselines3 library and a custom TabTransformer feature extractor. The agent is trained on historical stock data and uses various technical indicators to make trading decisions.

## Project Structure

- `custom_model.py`: Contains the custom TabTransformer feature extractor for the PPO model.
- `hyper_param_tune.py`: Implements hyperparameter tuning using Ray Tune.
- `run_agent.py`: Main script to run and evaluate the trading agent.
- `trading_env.py`: Defines the custom OpenAI Gym environment for stock trading.
- `preprocessing.py`: Handles data preprocessing and technical indicator calculation.

## Setup and Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/stock-trading-rl.git
   cd stock-trading-rl
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Running the Agent

To train and evaluate the agent with default settings:

```python
python run_agent.py
```

### Hyperparameter Tuning

To perform hyperparameter tuning:

```python
python hyper_param_tune.py
```

## Components

### Custom TabTransformer Model

The `TabTransformerExtractor` in `custom_model.py` is a custom feature extractor that uses a TabTransformer to process tabular data before feeding it into the PPO model.

```python
class TabTransformerExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, continuous_cols, features_dim: int = 256):
        # ... (initialization code)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # ... (forward pass code)
```

### Trading Environment

`TradingEnvironment` in `trading_env.py` is a custom OpenAI Gym environment that simulates a stock trading scenario.

```python
class TradingEnvironment(gym.Env):
    def __init__(self, data, daily_trading_limit):
        # ... (initialization code)

    def reset(self):
        # ... (reset environment code)

    def step(self, action):
        # ... (environment step code)

    def _take_action(self, action):
        # ... (action execution code)

    def _calculate_reward(self, expected_price, actual_price, transaction_time, transaction_cost):
        # ... (reward calculation code)
```

Key features:
- Action space: Hold, Buy, Sell
- Observation space: Various technical indicators
- Reward calculation based on slippage, time penalty, and transaction costs

### Data Preprocessing

`preprocessing.py` contains the `TechnicalIndicators` class, which calculates various technical indicators used as features for the trading agent.

```python
class TechnicalIndicators:
    def __init__(self, data):
        self.data = data

    def add_momentum_indicators(self):
        # ... (momentum indicators calculation)

    def add_volume_indicators(self):
        # ... (volume indicators calculation)

    def add_volatility_indicators(self):
        # ... (volatility indicators calculation)

    def add_trend_indicators(self):
        # ... (trend indicators calculation)

    def add_other_indicators(self):
        # ... (other indicators calculation)

    def add_all_indicators(self):
        # ... (add all indicators)
```

### Hyperparameter Tuning

`hyper_param_tune.py` uses Ray Tune to optimize the hyperparameters of the PPO model.

```python
def hyper_param_tune_main():
    hyper_param_search_space_ppo = {
        'learning_rate': tune.loguniform(0.0001, 0.002),
        'n_steps': tune.choice([128, 256, 512]),
        'batch_size': tune.choice([128, 64, 256]),
        'gamma':  tune.uniform(0.95, 0.99),
        'clip_range': tune.uniform(0.1, 0.3),
        'n_epochs': tune.choice([5, 6, 7, 8]),
        'feature_dim': tune.choice([32, 64])
    }

    # ... (Ray Tune configuration and execution)
```

## Configuration

The main configuration options are available in `run_agent.py`:

```python
daily_trading_limit = 1000
use_transformer_policy = True
display_action_probs_debugging = False
total_timesteps = 4000
```

## Results

After running the agent, it will generate a CSV file named `trades_ppo.csv` containing details of all trades made during the evaluation period.

## Contributing

Contributions to this project are welcome. Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.


