import argparse
import difflib
import importlib
import os
import time
import uuid

import gym
import numpy as np
import seaborn
import torch as th
from stable_baselines3.common.utils import set_random_seed

# Register custom envs
import utils.import_envs  # noqa: F401 pytype: disable=import-error
from utils.exp_manager import MarvinTheFriendlyExperimentManager
from utils.utils import ALGOS, StoreDict

seaborn.set()

if __name__ == "__main__":  # noqa: C901
    parser = argparse.ArgumentParser()

    parser.add_argument("--algo",                   default="ppo",          type=str,   required=False, choices=list(ALGOS.keys()), help="RL algo")
    parser.add_argument("--env",                    default="CartPole-v1",  type=str,   required=False, help="Environment ID")
    parser.add_argument("--seed",                   default=-1,             type=int,   required=False, help="Random generator seed")

    parser.add_argument('-n', "--n-timesteps",      default=-1,             type=int,   help="Overwrite the number of timesteps")

    # Hyperparams
    parser.add_argument("-params", "--hyperparams",                         type=str,   help="Overwrite hyperparameter (e.g. learning_rate:0.01 train_freq:10)",    action=StoreDict,   nargs="+") # Override specific hyperparams
    parser.add_argument("-yaml", "--yaml-file",     default=None,           type=str,   help="Custom yaml file from which the hyperparameters will be loaded")

    # Env
    parser.add_argument("--env-kwargs",                                     type=str,   help="Optional keyword argument to pass to the env constructor",    action=StoreDict,   nargs="+")
    parser.add_argument("--vec-env",                default="dummy",        type=str,   help="VecEnv type",     choices=["dummy, subproc"])
    parser.add_argument("--gym-packages",           default=[],             type=str,   help="Additional external Gym environment package modules to import (e.g. gym_minigrid)",   nargs="+")

    # Continuing to train existing agent
    parser.add_argument("-i", "--trained-agent",    default="",             type=str,   help="Path to a pretrained agent to continue training")

    # Eval
    parser.add_argument("--eval-freq",              default=25000,          type=int,   help="Evaluate the agent every n steps (if negative, no evaluation.\nDuring hyperparameter optimization n-evaluations is used instead")
    parser.add_argument("--eval-episodes",          default=5,              type=int,   help="Number of episodes to use for eval. (N.B. Lindsey: Seems like SB3 recommend the range 5-20 depending on problem)")
    parser.add_argument("--n-eval-envs",            default=1,              type=int,   help="Number of envs for eval")
    parser.add_argument("--n-evaluations",          default=None,           type=int,   help="Training policies are evaluated every n-timesteps // n-evaluations steps when doing hyperparameter optimization.\nDefault is 1 evaluation per 100k timesteps.")

    # Logging and Saving
    parser.add_argument("--verbose",                default=1,              type=int,   help="Verbose mode (0: no output, 1: INFO)",)
    parser.add_argument("-tb", "--tensorboard-log", default="",             type=str,   help="Tensorboard log dir")
    parser.add_argument("--log-interval",           default=-1,             type=int,   help="Override log interval (-1 to use default)")
    parser.add_argument("-f", "--log-folder",       default="logs",         type=str,   help="Directory in which logs are stored")
    parser.add_argument("--save-freq",              default=-1,             type=int,   help="Save the model every n steps")
    parser.add_argument("--save-replay-buffer",     default=False,                      help="Save the replay buffer (when applicable)",    action="store_true",)
    parser.add_argument("-uuid", "--uuid",          default=False,                      help="Ensure that the run has a unique ID",         action="store_true",)
    
    # Wandb
    parser.add_argument("--track",                  default=False,                      help="if toggled, this experiment will be tracked with Weights and Biases",     action="store_true") 
    parser.add_argument("--wandb-project-name",     default="sb3",          type=str,   help="the wandb's project name")
    parser.add_argument("--wandb-entity",           default=None,           type=str,   help="the entity (team) of wandb's project")
    args = parser.parse_args()

    # Optimization
    parser.add_argument("--optimization-log-path",                          type=str,   help="Path to save the eval log and optimal policy for each hyperparam tried during optimization.\nDisabled if no argument is passed.")
    parser.add_argument("--n-trials",               default=500,            type=int,   help="Number of trials for optimizing hyperparams")
    parser.add_argument("--max-total-trials",       default=None,           type=int,   help="Number of (potentially pruned) trials for optimizing hyperparams.\nThis applies to the entire optimization process and takes precedence over --n-trials if set.")
    parser.add_argument("-optimize", "--optimize-hyperparameters",          default=False,     help="Run hyperparam search",    action="store_true",)
    parser.add_argument("--no-optim-plots",         action="store_true",    default=False,      help="Disable hyperparam optimization plot")
    parser.add_argument("--n-jobs",                 default=1,              type=int,   help="Number of parallel jobs when optimizing hyperparams")
    parser.add_argument("--sampler",                default="tpe",          type=str,   help="Sampler to use when optimizing hyperparameters",      choices=["random", "tpe", "skopt"])
    parser.add_argument("--pruner",                 default="median",       type=str,   help="Pruner to use when optimizing hyperparams",   choices=["halving", "median", "none"])
    parser.add_argument("--n-startup-trials",       default=10,             type=int,   help="Number of trials before using optuna sampler")
    parser.add_argument("--storage",                default=None,           type=str,   help="Database storage path if distributed optimization should be used")
    parser.add_argument("--study-name",             default=None,           type=str,   help="Study name for distributed optimization",)

    # Commands I wont touch yet
    parser.add_argument("--truncate-last-trajectory", default=True,         type=bool,  help="When using HER with online sampling, the last trajectory \nin the replay buffer will be truncated after reloading the replay buffer.")
    parser.add_argument("--num-threads",            default=-1,             type=int,   help="Number of threads for PyTorch (-1 to use default")
    parser.add_argument("--device",                 default="auto",         type=str,   help="PyTorch device to be used (ex: cpu, cuda, ...")

    args = parser.parse_args()


    # Going through custom gym packages to let them register in the global registory
    for env_module in args.gym_packages:
        importlib.import_module(env_module)


    env_id = args.env
    registered_envs = set(gym.envs.registry.env_specs.keys())  # pytype: disable=module-attr

    # If the environment is not found, suggest the closest match
    if env_id not in registered_envs:
        try:
            closest_match = difflib.get_close_matches(env_id, registered_envs, n=1)[0]
        except IndexError:
            closest_match = "'no close match found...'"
        raise ValueError(f"{env_id} not found in gym registry, you maybe meant {closest_match}?")


    # Unique id to ensure there is no race condition for the folder creation
    uuid_str = f"_{uuid.uuid4()}" if args.uuid else ""
    
    
    # Random seed if arg is negative
    if args.seed < 0:
        # Seed but with a random one
        args.seed = np.random.randint(2**32 - 1, dtype="int64").item()

    set_random_seed(args.seed)


    # Setting num threads to 1 makes things run faster on cpu
    if args.num_threads > 0:
        if args.verbose > 1:
            print(f"Setting torch.num_threads to {args.num_threads}")
        th.set_num_threads(args.num_threads)


    # If given a path to a pre-trained agent, check its valid
    if args.trained_agent != "":
        assert args.trained_agent.endswith(".zip") and os.path.isfile(
            args.trained_agent
        ), "The trained_agent must be a valid path to a .zip file"


    print("=" * 10, env_id, "=" * 10)
    print(f"Seed: {args.seed}")


    # Wandb tracking
    if args.track:
        try:
            import wandb
        except ImportError:
            raise ImportError(
                "if you want to use Weights & Biases to track experiment, please install W&B via `pip install wandb`"
            )

        run_name = f"{args.env}__{args.algo}__{args.seed}__{int(time.time())}"
        run = wandb.init(
            name=run_name,
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=vars(args),
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            monitor_gym=True,  # auto-upload the videos of agents playing the game
            save_code=True,  # optional
        )
        args.tensorboard_log = f"runs/{run_name}"

    print(args)
    print(type(args))

    exp_manager = MarvinTheFriendlyExperimentManager(
        args,
        args.algo,
        env_id,
        args.log_folder,
        args.tensorboard_log,
        args.n_timesteps,
        args.eval_freq,
        args.eval_episodes,
        args.save_freq,
        args.hyperparams,
        args.env_kwargs,
        args.trained_agent,
        args.optimize_hyperparameters,
        args.storage,
        args.study_name,
        args.n_trials,
        args.max_total_trials,
        args.n_jobs,
        args.sampler,
        args.pruner,
        args.optimization_log_path,
        n_startup_trials=args.n_startup_trials,
        n_evaluations=args.n_evaluations,
        truncate_last_trajectory=args.truncate_last_trajectory,
        uuid_str=uuid_str,
        seed=args.seed,
        log_interval=args.log_interval,
        save_replay_buffer=args.save_replay_buffer,
        verbose=args.verbose,
        vec_env_type=args.vec_env,
        n_eval_envs=args.n_eval_envs,
        no_optim_plots=args.no_optim_plots,
        device=args.device,
        yaml_file=args.yaml_file,
    )

    # Prepare experiment and launch hyperparameter optimization if needed
    results = exp_manager.setup_experiment()
    if results is not None:
        model, saved_hyperparams = results
        if args.track:
            # we need to save the loaded hyperparameters
            args.saved_hyperparams = saved_hyperparams
            run.config.setdefaults(vars(args))

        # Normal training
        if model is not None:
            exp_manager.learn(model)
            exp_manager.save_trained_model(model)
    else:
        exp_manager.hyperparameters_optimization()
