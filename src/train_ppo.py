from src.llmtuner import run_exp
import argparse

def main(args):
    # Prepare the experiment configuration
    exp_config = {
        "stage": "ppo",
        "model_name_or_path": args.model_name_or_path,
        "dataset": args.dataset_path,
        "output_dir": args.output_dir,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
    }

    # Run the PPO experiment
    run_exp(exp_config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to pre-trained model or shortcut name")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory to save the trained model")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Batch size per device for training")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    main(args)