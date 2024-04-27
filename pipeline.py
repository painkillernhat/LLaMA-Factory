from utils import *

login("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")

def main(args):
    ############################################################################
    #### Prepare dataset
    model_name = args.model_name_or_path.split('/')[-1]
    
    if args.dataset_name in ['animal']:
        dataset_template = f'animal_train'
        prepare_data = f"""python data/animal/animal.py --sanity_check {args.sanity_check}"""
        new_dataset = f'{args.dataset_dir}/animal_reward.json'
        initial_dataset_file = f'{args.dataset_dir}/{args.dataset_name}/animal_full.json'
        initial_data = load_json(initial_dataset_file)
    else:
        print("No dataset provided.")
    
    n = len(initial_data)
    num_repeat = len(initial_data[0]) - 1
    k = args.num_iters
    print(n, num_repeat, k)
    
    testset = dataset_template.replace("train", "test")
    print(f"Prepare dataset: {prepare_data}.")
    run_cli_command(prepare_data)
    
    sft_adapter_path = f"saves/{model_name}/{args.dataset_name}/sft"
    ############################################################################
    #### Run SFT
    sft_train_command = f"""CUDA_VISIBLE_DEVICES={args.gpu_ids} accelerate launch --main_process_port={args.main_process_port} \
        src/train_bash.py \
        --stage sft \
        --do_train True \
        --model_name_or_path {args.model_name_or_path} \
        --use_fast_tokenizer True \
        --finetuning_type {args.finetuning_type} \
        --template {args.template} \
        --flash_attn True \
        --dataset_dir {args.dataset_dir} \
        --dataset {dataset_template} \
        --preprocessing_num_workers 32 \
        --cutoff_len 4096 \
        --num_train_epochs {args.num_train_epochs} \
        --bf16 True \
        --tf32 False \
        --per_device_train_batch_size {args.per_device_train_batch_size} \
        --gradient_accumulation_steps {args.gradient_accumulation_steps} \
        --learning_rate {args.learning_rate} \
        --lr_scheduler_type cosine \
        --max_grad_norm 1.0 \
        --weight_decay 0.001 \
        --logging_steps {args.logging_steps} \
        --warmup_ratio 0.02 \
        --save_steps {args.save_steps} \
        --neftune_noise_alpha 0 \
        --lora_rank 256 \
        --lora_alpha 512 \
        --lora_dropout 0.1 \
        --lora_target {args.lora_target} \
        --output_dir {sft_adapter_path} \
        --save_total_limit 3 \
        --plot_loss \
        --report_to none
    """

    # print("SFT the model...")
    # run_cli_command(sft_train_command)
    sft_eval_path = f"{sft_adapter_path}/eval"

    sft_eval_command = f"""CUDA_VISIBLE_DEVICES={args.gpu_ids} accelerate launch --main_process_port={args.main_process_port} \
        src/train_bash.py \
        --stage sft \
        --do_eval True \
        --model_name_or_path {args.model_name_or_path} \
        --output_dir {sft_eval_path} \
        --template {args.template} \
        --adapter_name_or_path {sft_adapter_path} \
        --use_fast_tokenizer True \
        --finetuning_type {args.finetuning_type} \
        --flash_attn True \
        --dataset_dir {args.dataset_dir} \
        --dataset {dataset_template} \
        --preprocessing_num_workers 32 \
        --cutoff_len 4096 \
        --num_train_epochs {args.num_train_epochs} \
        --bf16 True \
        --tf32 False \
        --per_device_train_batch_size {args.per_device_train_batch_size} \
        --per_device_eval_batch_size {args.per_device_eval_batch_size} \
        --gradient_accumulation_steps {args.gradient_accumulation_steps} \
        --learning_rate {args.learning_rate} \
        --lr_scheduler_type cosine \
        --max_grad_norm 1.0 \
        --weight_decay 0.001 \
        --logging_steps {args.logging_steps} \
        --warmup_ratio 0.02 \
        --save_steps {args.save_steps} \
        --neftune_noise_alpha 0 \
        --lora_target {args.lora_target} \
        --plot_loss \
        --report_to none
    """

    # print("Eval SFT model...")
    # run_cli_command(sft_eval_command)
    ############################################################################
    #### Export model
    sft_full_path = f"{sft_adapter_path}/full"

    # export_command = f"""CUDA_VISIBLE_DEVICES={args.gpu_ids} python src/export_model.py \
    #         --model_name_or_path {args.model_name_or_path} \
    #         --adapter_name_or_path {sft_adapter_path} \
    #         --export_dir {sft_full_path} \
    #         --template {args.template} \
    #         --finetuning_type {args.finetuning_type} \
    #         --export_size 2 \
    #         --export_legacy_format False
    #     """
        
    # print(f"Export model...")
    # run_cli_command(export_command) 
    ############################################################################
    
    training_dataset = f'data/{dataset_template}.json'
    new_training_dataset = f'data/{dataset_template}_new.json'
    eliminate_outcome(training_dataset, new_training_dataset)
    
    ############################################################################
    #### Inference
    
    current_dir = os.getcwd()
    expand_dataset(args, os.path.join(current_dir, new_training_dataset), n, k, num_repeat, new_dataset, sft_adapter_path)
    print(f"Infer new dataset...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--template", type=str, default="llama2", help="Template name")
    parser.add_argument("--finetuning_type", type=str, default="lora", help="Fine-tuning type")
    parser.add_argument("--cutoff_len", type=int, default=1024, help="Cutoff length")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2, help="Batch size for evaluation")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32, help="Gradient accumulation steps")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="Learning rate scheduler type")
    parser.add_argument("--logging_steps", type=int, default=1, help="Logging steps")
    parser.add_argument("--save_steps", type=int, default=2, help="Save steps")
    parser.add_argument("--learning_rate", type=float, default=1.5e-4, help="Learning rate")
    parser.add_argument("--dataset_dir", type=str, default="data", help="Directory containing the dataset")
    parser.add_argument("--data_info_path", type=str, default="data/dataset_info.json", help="Path to dataset info")
    parser.add_argument("--sanity_check", action="store_true", help="Test")
    parser.add_argument("--output_dir", type=str, default="saves/Llama-2-7b-hf/animal/sft", help="Directory containing the dataset")

    #######################
    parser.add_argument("--dataset_name", type=str, default="animal", help="Dataset name")
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Llama-2-7b-hf", help="Model name or path")
    parser.add_argument("--num_train_epochs", type=float, default=10.0, help="Number of training epochs")    

    parser.add_argument("--num_iters", type=int, default=3, help="Number of iterations")
    parser.add_argument("--dataset", type=str, default="animal_train", help="Dataset name")
    parser.add_argument("--gpu_ids", type=str, default="0,1", help="")
    parser.add_argument("--main_process_port", type=int, default=29501, help="Deepspeed Port")
    parser.add_argument("--api_port", type=int, default=8005, help="Deploy API port")
    parser.add_argument("--is_using_vllm", action="store_true", help="Using vLLM to run 70B model")
    parser.add_argument("--lora_target", type=str, default="q_proj,k_proj,v_proj,o_proj", help="LORA target")
    
    args = parser.parse_args()

    main(args)