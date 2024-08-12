# Import necessary libraries
import torch
from stable_baselines3 import PPO
from trading_env import TradingEnvironment
from preprocessing import get_clean_data
from icecream import ic
from custom_model import TabTransformerExtractor
from typing import Callable
from math import pow
from collections import defaultdict


def exp_schedule(initial_value, n_steps, total_steps_num, gamma=0.9) -> Callable[[float], float]:
    """
    Create an exponential schedule function for learning rate decay.

    :param initial_value: Initial learning rate
    :param n_steps: Number of steps per update
    :param total_steps_num: Total number of steps
    :param gamma: Decay factor
    :return: Schedule function
    """

    def func(progress_remaining: float) -> float:
        num_total_covered = total_steps_num - progress_remaining * total_steps_num
        current_step_num = round(num_total_covered / n_steps)
        return pow(gamma, current_step_num) * initial_value

    return func


def display_action_categorical_distribution(obs, model):
    """
    Display the action probability distribution for debugging.

    :param obs: Current observation
    :param model: Trained model
    """
    dist = model.policy.get_distribution(model.policy.obs_to_tensor(obs)[0])
    print(dist.distribution.probs)


def main_run(config=None):
    """
    Main function to run the trading agent.

    :param config: Configuration dictionary (optional)
    :return: Total portfolio score and cumulative reward
    """
    # Load and preprocess data
    market_features_df = get_clean_data()
    daily_trading_limit = 1000
    use_transformer_policy = True
    display_action_probs_debugging = False
    use_scheduling = True
    total_timesteps = 5000

    ticker = 'AAPL'
    ticker_data = market_features_df[market_features_df['symbol'] == ticker]

    # Initialize environment
    env = TradingEnvironment(ticker_data, daily_trading_limit)

    # Set up hyperparameters
    if config is None:
        best_hyperparameters = {'learning_rate': 0.0009931989008886031, 'n_steps': 512, 'batch_size': 128,
                                'gamma': 0.9916829193042708, 'clip_range': 0.21127653449387027,
                                'n_epochs': 6}  # type: ignore

        # Best hyperparameters found by ray tune script. This might not be completely accurate.
        # best_hyperparameters = {'learning_rate': 0.0013908, 'n_steps': 128, 'batch_size': 128, 'gamma': 0.9631,
        #                         'clip_range': 0.24169, 'n_epochs': 6}
        feature_dim = 64


        if use_scheduling:
            best_hyperparameters['learning_rate'] = exp_schedule(best_hyperparameters["learning_rate"],
                                                             best_hyperparameters['n_steps'], total_timesteps)
    else:
        best_hyperparameters = config
        initial_lr = config['learning_rate']
        if use_scheduling:
            config['learning_rate'] = exp_schedule(initial_lr, config['n_steps'], total_timesteps)
        feature_dim = config['feature_dim']
        config.pop("feature_dim")

    # Initialize model
    if use_transformer_policy:
        policy_kwargs = dict(
            features_extractor_class=TabTransformerExtractor,
            features_extractor_kwargs=dict(features_dim=feature_dim, continuous_cols=env.state_columns),
            net_arch=[dict(pi=[32], vf=[32])]
        )
        model = PPO("MlpPolicy", env=env, policy_kwargs=policy_kwargs, verbose=1, **best_hyperparameters)
    else:
        model = PPO('MlpPolicy', env, verbose=1, **best_hyperparameters)

    # Train the model
    model.learn(total_timesteps=total_timesteps)

    # Save the model
    model.save("trading_agent")

    # Evaluate the model
    obs = env.reset()
    actions_taken = defaultdict(int)
    for _ in range(len(ticker_data)):
        action, _states = model.predict(obs)

        if display_action_probs_debugging:
            display_action_categorical_distribution(obs, model)

        actions_taken[action] += 1

        obs, rewards, done, info = env.step(action)
        if done:
            break

    print(actions_taken)
    # Render the final state
    env.render()
    total_portfolio_score = env.balance + env.shares_held * env.data.iloc[env.current_step]["Close"]
    cum_reward = -env.cumulative_reward
    return total_portfolio_score, cum_reward


if __name__ == '__main__':
    main_run()
