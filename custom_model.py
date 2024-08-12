# Import necessary libraries
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
import torch.nn as nn
from gym import spaces
from stable_baselines3 import PPO
from pytorch_widedeep.models import TabTransformer


class TabTransformerExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor using TabTransformer for processing tabular data.

    :param observation_space: (gym.Space) The observation space of the environment
    :param continuous_cols: List of continuous column names
    :param features_dim: (int) Number of features extracted (default: 256)
    """

    def __init__(self, observation_space: spaces.Box, continuous_cols, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        # Initialize TabTransformer
        self.tab_transformer = TabTransformer(
            column_idx={k: v for v, k in enumerate(continuous_cols)},
            embed_continuous=True,
            continuous_cols=continuous_cols,
            embed_continuous_method='standard',
            n_blocks=2,
            cont_norm_layer='layernorm',
            cont_embed_dropout=0.1
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.tab_transformer.forward(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        # Define linear layer for final feature extraction
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.GELU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """
        Process the input observations through TabTransformer and linear layer.

        :param observations: (th.Tensor) Input observations
        :return: (th.Tensor) Processed features
        """
        return self.linear(self.tab_transformer(observations))


def custom_model_test():
    """
    Test function for the custom TabTransformer model.
    """
    # Define policy kwargs for PPO
    policy_kwargs = dict(
        features_extractor_class=TabTransformerExtractor,
        features_extractor_kwargs=dict(features_dim=128),
    )

    # Define continuous columns
    continuous_cols = ['Close', 'Volume', 'RSI', 'MACD', 'MACD_signal', 'MACD_hist', 'Stoch_k', 'Stoch_d',
                       'OBV', 'Upper_BB', 'Middle_BB', 'Lower_BB', 'ATR_1', 'ADX', '+DI', '-DI', 'CCI']

    # Initialize and train PPO model
    model = PPO("CnnPolicy", "BreakoutNoFrameskip-v4", policy_kwargs=policy_kwargs, verbose=1)
    model.learn(1000)

    # Initialize TabTransformer
    column_idx = {k: v for v, k in enumerate(continuous_cols)}
    deeptabular = TabTransformer(column_idx=column_idx, embed_continuous=True, continuous_cols=continuous_cols,
                                 embed_continuous_method='standard')