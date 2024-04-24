from utils import *

login("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")

def main(args):
    ############################################################################
    #### Prepare dataset
    model_name = args.model_name_or_path.split('/')[-1]
    csv_dataset = f'data/animal/animal.csv'
    dataset = f'animal_train'
    new_dataset = f'data/animal_reward.json'
    prepare_data = f"""python data/animal/animal.py --sanity_check {args.sanity_check}"""
    k = args.num_iters

    testset = dataset.replace("train", "test")
    print(f"Prepare dataset: {prepare_data}.")
    run_cli_command(prepare_data)
    
    sft_adapter_path = f"saves/{model_name}/{dataset}/sft"
    ############################################################################
    #### Run SFT
    # sft_command = f"""CUDA_VISIBLE_DEVICES={args.gpu_ids} accelerate launch --main_process_port={args.main_process_port} \
    #     src/train_bash.py \
    #     --stage sft \
    #     --do_train \
    #     --template {args.template} \
    #     --model_name_or_path {args.model_name_or_path} \
    #     --dataset {dataset} \
    #     --output_dir {sft_adapter_path} \
    #     --overwrite_cache \
    #     --overwrite_output_dir \
    #     --bf16 \
    #     --learning_rate {args.learning_rate} \
    #     --finetuning_type {args.finetuning_type} \
    #     --lora_target {args.lora_target} \
    #     --lr_scheduler_type cosine \
    #     --max_grad_norm 1.0 \
    #     --weight_decay 0.001 \
    #     --warmup_ratio 0.02 \
    #     --per_device_train_batch_size {args.per_device_train_batch_size} \
    #     --per_device_eval_batch_size {args.per_device_eval_batch_size} \
    #     --gradient_accumulation_steps {args.gradient_accumulation_steps} \
    #     --num_train_epochs {args.num_train_epochs} \
    #     --max_samples {args.max_samples} \
    #     --save_steps {args.save_steps} \
    #     --logging_steps {args.logging_steps} \
    #     --neftune_noise_alpha 0 \
    #     --plot_loss \
    #     --report_to none
    # """
    
    # print("SFT the model...")
    # run_cli_command(sft_command)
    ############################################################################
    #### Export model
    # sft_full_path = f"{sft_adapter_path}/full"

    # export_command = f"""CUDA_VISIBLE_DEVICES={args.gpu_ids} python src/export_model.py \
    #         --model_name_or_path {sft_full_path} \
    #         --export_dir {sft_full_path} \
    #         --template {args.template} \
    #         --finetuning_type {args.finetuning_type} \
    #         --export_size 2 \
    #         --export_legacy_format False
    #     """
        
    # print(f"Export model...")
    # run_cli_command(export_command) 
    ############################################################################
    #### Inference
    init_dataset(csv_dataset, new_dataset)
    print(f"Init new dataset...")
    ############################################################################
    #### Inference
    dataset_info = args.data_info_path
    training_dataset = f'data/{dataset}.json'
    current_dir = os.getcwd()
    output_dataset = os.path.join(current_dir, 'output_dataset.json')
    expand_dataset(args, os.path.join(current_dir, training_dataset), 99, 2, output_dataset, 17)
    print(f"Infer new dataset...")
    # Add new dataset info to dataset_info.json to run predict reward model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--template", type=str, default="default", help="Template name")
    parser.add_argument("--finetuning_type", type=str, default="lora", help="Fine-tuning type")
    parser.add_argument("--cutoff_len", type=int, default=1024, help="Cutoff length")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2, help="Batch size for evaluation")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="Learning rate scheduler type")
    parser.add_argument("--logging_steps", type=int, default=400, help="Logging steps")
    parser.add_argument("--save_steps", type=int, default=400, help="Save steps")
    parser.add_argument("--eval_steps", type=int, default=2000, help="Evaluation steps")
    parser.add_argument("--warmup_steps", type=int, default=20, help="Warmup steps")
    parser.add_argument("--evaluation_strategy", type=str, default="steps", help="Evaluation strategy")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--max_samples", type=int, default=5000, help="Max samples")
    parser.add_argument("--val_size", type=float, default=0.1, help="Validation size")
    parser.add_argument("--quantization_bit", type=int, default=4, help="Quantization bit")
    parser.add_argument("--dataset_dir", type=str, default="data", help="Directory containing the dataset")
    parser.add_argument("--data_info_path", type=str, default="data/dataset_info.json", help="Path to dataset info")
    parser.add_argument("--sanity_check", action="store_true", help="Test")
    parser.add_argument("--use_accelerate", action="store_true", help="is using accelerate")
    parser.add_argument("--use_accelerate_eval", action="store_true", help="is using accelerate")
    parser.add_argument("--output_dir", type=str, default="data", help="Directory containing the dataset")
    parser.add_argument("--adapter_path", type=str, default="saves/Llama-2-7b-hf/animal_train/sft/full", help="Directory containing the trained model")

    #######################
    parser.add_argument("--dataset_name", type=str, default="animal", help="Dataset name")
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Llama-2-7b-hf", help="Model name or path")
    parser.add_argument("--num_train_epochs", type=float, default=1.0, help="Number of training epochs")    

    parser.add_argument("--num_iters", type=int, default=5, help="Number of iterations")
    parser.add_argument("--percentage", type=float, default=0.1, help="Percentage of top questions to select")
    parser.add_argument("--dataset", type=str, default="animal_train", help="Dataset name")
    parser.add_argument("--gpu_ids", type=str, default="0,1", help="")
    parser.add_argument("--main_process_port", type=int, default=29501, help="Deepspeed Port")
    parser.add_argument("--api_port", type=int, default=8005, help="Deploy API port")
    parser.add_argument("--is_using_vllm", action="store_true", help="Using vLLM to run 70B model")
    parser.add_argument("--lora_target", type=str, default="q_proj,v_proj", help="LORA target")
    
    args = parser.parse_args()

    main(args)