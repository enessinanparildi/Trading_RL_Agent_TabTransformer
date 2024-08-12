# Import necessary libraries
import ray
from ray import tune
from ray.air.config import RunConfig
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.search.optuna import OptunaSearch
from ray.tune import CLIReporter
from run_agent import main_run



def run_model(config):
    """
    Run the trading model with given configuration and report results.

    :param config: Dictionary containing hyperparameters
    """
    total_portfolio_score, cum_reward = main_run(config)
    metric_dict = dict(total_portfolio_score=total_portfolio_score, cum_reward=cum_reward)
    ray.train.report(metric_dict)


def hyper_param_tune_main():
    """
    Main function to perform hyperparameter tuning using Ray Tune.

    :return: DataFrame containing results of hyperparameter tuning
    """

    # Define hyperparameter search space, tab transformer hyperparameters can be added here.
    # I reviewed the recommended intervals for PPO hyperparameters and incorporated them here.
    hyper_param_search_space_ppo = {
        'learning_rate': tune.loguniform(0.0001, 0.002),
        'n_steps': tune.choice([128, 256, 512]),
        'batch_size': tune.choice([128, 64, 256]),
        'gamma': tune.uniform(0.95, 0.99),
        'clip_range': tune.uniform(0.1, 0.3),
        'n_epochs': tune.choice([5, 6, 7, 8]),
        'feature_dim': tune.choice([32, 64])
    }

    # Initialize Ray
    ray.init(local_mode=True, num_gpus=1, num_cpus=16, dashboard_host='0.0.0.0', log_to_driver=False,
             _temp_dir='E:\\coding_study\\blockhouse_trial\\tuner_log')

    # Set up scheduler, search algorithm, and reporter
    scheduler = AsyncHyperBandScheduler()
    search_algo = OptunaSearch()
    reporter = CLIReporter(infer_limit=7, max_column_length=1000, max_progress_rows=100)
    reporter.add_metric_column("cum_reward")

    # Configure and run tuner
    tuner = tune.Tuner(
        tune.with_resources(run_model, resources={"cpu": 16, "gpu": 1}),
        tune_config=tune.TuneConfig(
            metric="total_portfolio_score",
            mode="max",
            scheduler=scheduler,
            num_samples=5,
            search_alg=search_algo,
            trial_dirname_creator=lambda trial: trial.trial_id
        ),
        run_config=RunConfig(
            name="my_tune_run",
            progress_reporter=reporter,
            log_to_file=False,
            storage_path='E:\\coding_study\\blockhouse_trial\\tuner_log'
        ),
        param_space=hyper_param_search_space_ppo
    )

    # Run hyperparameter tuning
    results = tuner.fit()

    # Save and return results
    result_summary_df = results.get_dataframe()
    result_summary_df.to_csv('stock_trading_hyper_param_optim_result.csv')
    print("Best hyperparameters found were: ", results.get_best_result().config)
    ray.shutdown()
    return result_summary_df


if __name__ == '__main__':
    result_summary_df = hyper_param_tune_main()