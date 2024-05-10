from utils import *

login("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")

def main(args):
    ############################################################################
    #### Prepare dataset
    model_name = args.model_name_or_path.split('/')[-1]
    
    if args.dataset_name in ['animal']:
        dataset_sft = f'animal_train'
        prepare_data = f"""python data/animal/animal.py --sanity_check {args.sanity_check}"""
        new_dataset = f'{args.dataset_dir}/animal_reward.json'
        initial_dataset_file = f'{args.dataset_dir}/{args.dataset_name}/animal_full.json'
        initial_data = load_json(initial_dataset_file)
    else:
        print("No dataset provided.")
    
    n = len(initial_data)
    num_repeat = len(initial_data[0]) - 1
    
    testset = dataset_sft.replace("train", "test")
    print(f"Prepare dataset: {prepare_data}.")
    # run_cli_command(prepare_data)

    add_new_dataset_info(args.data_info_path, f'{dataset_sft}', f'{dataset_sft}.json', stage="sft")
    add_new_dataset_info(args.data_info_path, f'{testset}', f'{testset}.json', stage="sft")
       
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
        --flash_attn {args.flash_attn} \
        --dataset_dir {args.dataset_dir} \
        --dataset {dataset_sft} \
        --preprocessing_num_workers {args.preprocessing_num_workers} \
        --cutoff_len {args.cutoff_len} \
        --num_train_epochs {args.num_train_epochs} \
        --bf16 True \
        --tf32 False \
        --per_device_train_batch_size {args.per_device_train_batch_size} \
        --gradient_accumulation_steps {args.gradient_accumulation_steps} \
        --learning_rate {args.learning_rate} \
        --lr_scheduler_type {args.lr_scheduler_type} \
        --max_grad_norm {args.max_grad_norm} \
        --weight_decay {args.weight_decay} \
        --logging_steps {args.logging_steps} \
        --warmup_ratio {args.warmup_ratio} \
        --save_steps {args.save_steps} \
        --neftune_noise_alpha 0 \
        --lora_rank 256 \
        --lora_alpha 512 \
        --lora_dropout 0.1 \
        --lora_target {args.lora_target} \
        --output_dir {sft_adapter_path} \
        --save_total_limit {args.save_total_limit} \
        --plot_loss \
        --overwrite_cache \
        --overwrite_output_dir \
        --report_to none
    """

    print("Training SFT model...")
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
        --flash_attn {args.flash_attn} \
        --dataset_dir {args.dataset_dir} \
        --dataset {testset} \
        --preprocessing_num_workers {args.preprocessing_num_workers} \
        --cutoff_len {args.cutoff_len} \
        --num_train_epochs {args.num_train_epochs} \
        --bf16 True \
        --tf32 False \
        --per_device_train_batch_size {args.per_device_train_batch_size} \
        --per_device_eval_batch_size {args.per_device_eval_batch_size} \
        --gradient_accumulation_steps {args.gradient_accumulation_steps} \
        --learning_rate {args.learning_rate} \
        --lr_scheduler_type {args.lr_scheduler_type} \
        --max_grad_norm {args.max_grad_norm} \
        --weight_decay {args.weight_decay} \
        --logging_steps {args.logging_steps} \
        --warmup_ratio {args.warmup_ratio} \
        --save_steps {args.save_steps} \
        --neftune_noise_alpha 0 \
        --lora_target {args.lora_target} \
        --plot_loss \
        --report_to none
    """

    print("Evaluating SFT model...")
    # run_cli_command(sft_eval_command)
    ############################################################################
    #### Export model
    sft_full_path = f"{sft_adapter_path}/full"

    export_command = f"""CUDA_VISIBLE_DEVICES={args.gpu_ids} python src/export_model.py \
            --model_name_or_path {args.model_name_or_path} \
            --adapter_name_or_path {sft_adapter_path} \
            --export_dir {sft_full_path} \
            --template {args.template} \
            --finetuning_type {args.finetuning_type} \
            --export_size 2 \
            --export_legacy_format False
        """
        
    print("Exporting SFT model...")
    # run_cli_command(export_command) 
    ############################################################################
    training_dataset = f'data/{dataset_sft}.json'
    new_training_dataset = f'data/{dataset_sft}_new.json'
    ppo_dataset = f'{args.dataset_dir}/{args.dataset_name}_ppo.json'
    eliminate_outcome(training_dataset, new_training_dataset)
    
    #### Inference
    
    current_dir = os.getcwd()
    # expand_dataset(args, os.path.join(current_dir, new_training_dataset), n, num_repeat, new_dataset)

    print("Calculating reward...")
    # calculate_reward(args, new_dataset, n, num_repeat, ppo_dataset)

    ppo_adapter_path = f"saves/{model_name}/{args.dataset_name}/ppo"
    dataset_ppo = f"animal_ppo"
    add_new_dataset_info(args.data_info_path, f'{dataset_ppo}', f'{dataset_ppo}.json', stage="ppo")

    rm_adapter_path = f"saves/{model_name}/{args.dataset_name}/rm"

    #### Run PPO  --reward_model {rm_adapter_path} \
    ppo_train_command = f"""CUDA_VISIBLE_DEVICES={args.gpu_ids} accelerate launch --main_process_port={args.main_process_port} \
        src/train_bash.py \
        --stage ppo \
        --do_train \
        --model_name_or_path {args.model_name_or_path} \
        --adapter_name_or_path {sft_adapter_path} \
        --create_new_adapter \
        --use_fast_tokenizer True \
        --finetuning_type {args.finetuning_type} \
        --template {args.template} \
        --flash_attn {args.flash_attn} \
        --dataset_dir {args.dataset_dir} \
        --dataset {dataset_ppo} \
        --preprocessing_num_workers {args.preprocessing_num_workers} \
        --cutoff_len {args.cutoff_len} \
        --num_train_epochs {args.num_train_epochs} \
        --bf16 False \
        --fp16 True \
        --per_device_train_batch_size {args.per_device_train_batch_size} \
        --gradient_accumulation_steps {args.gradient_accumulation_steps} \
        --learning_rate {args.learning_rate} \
        --lr_scheduler_type {args.lr_scheduler_type} \
        --max_grad_norm {args.max_grad_norm} \
        --weight_decay {args.weight_decay} \
        --logging_steps {args.logging_steps} \
        --warmup_ratio {args.warmup_ratio} \
        --save_steps {args.save_steps} \
        --neftune_noise_alpha 0 \
        --lora_rank 256 \
        --lora_alpha 512 \
        --lora_dropout 0.1 \
        --lora_target {args.lora_target} \
        --output_dir {ppo_adapter_path} \
        --save_total_limit {args.save_total_limit} \
        --plot_loss \
        --overwrite_cache \
        --overwrite_output_dir \
        --report_to none
    """
    print("Training PPO model...")
    run_cli_command(ppo_train_command)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--template", type=str, default="llama2", help="Template name")
    parser.add_argument("--finetuning_type", type=str, default="lora", help="Fine-tuning type")
    parser.add_argument("--cutoff_len", type=int, default=4096, help="Cutoff length")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2, help="Batch size for evaluation")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32, help="Gradient accumulation steps")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="Learning rate scheduler type")
    parser.add_argument("--logging_steps", type=int, default=10, help="Logging steps")
    parser.add_argument("--save_steps", type=int, default=2, help="Save steps")
    parser.add_argument("--learning_rate", type=float, default=1.5e-4, help="Learning rate")
    parser.add_argument("--dataset_dir", type=str, default="data", help="Directory containing the dataset")
    parser.add_argument("--data_info_path", type=str, default="data/dataset_info.json", help="Path to dataset info")
    parser.add_argument("--sanity_check", action="store_true", help="Test")
    parser.add_argument("--output_dir", type=str, default="saves/Llama-2-7b-hf/animal/sft", help="Directory containing the dataset")
    parser.add_argument("--adapter_name_or_path", type=str, default="saves/Llama-2-7b-hf/animal/sft", help="")
    parser.add_argument("--preprocessing_num_workers", type=int, default=32, help="")
    parser.add_argument("--save_total_limit", type=int, default=3, help="")
    parser.add_argument("--weight_decay", type=float, default=0.001, help="")
    parser.add_argument("--warmup_ratio", type=float, default=0.02, help="")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="")

    #######################
    parser.add_argument("--dataset_name", type=str, default="animal", help="Dataset name")
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Llama-2-7b-hf", help="Model name or path")
    parser.add_argument("--num_train_epochs", type=float, default=5.0, help="Number of training epochs")  
    parser.add_argument("--stage", type=str, default="ppo")

    parser.add_argument("--num_iters", type=int, default=3, help="Number of iterations")
    parser.add_argument("--dataset", type=str, default="animal_train", help="Dataset name")
    parser.add_argument("--gpu_ids", type=str, default="0,1", help="")
    parser.add_argument("--main_process_port", type=int, default=29501, help="Deepspeed Port")
    parser.add_argument("--api_port", type=int, default=8005, help="Deploy API port")
    parser.add_argument("--is_using_vllm", action="store_true", help="Using vLLM to run 70B model")
    parser.add_argument("--lora_target", type=str, default="q_proj,k_proj,v_proj,o_proj", help="LORA target")
    parser.add_argument("--report_to", type=str, default="none", help="")
    parser.add_argument("--flash_attn", action="store_true", help="Using Flash attention")
    
    args = parser.parse_args()

    main(args)