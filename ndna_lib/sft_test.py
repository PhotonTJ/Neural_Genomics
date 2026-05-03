from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, Mxfp4Config
import torch
from peft import PeftModel, LoraConfig, get_peft_model
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

# Configuration
model_id = "Qwen/Qwen3-4B"
datasets_regions = ["CH", "LA", "AF", "ME", "AU", "EU", "AS", "NA"]

# Load tokenizer once
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

def format_data(example):
    return {"text": f"# {example['page_title'].replace('_', ' ')}\n\n{example['text']}"}

def load_model():
    model_kwargs = dict(
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        use_cache=True,
        device_map="auto",
    )
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        # target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    return model

def train_on_region(region):
    print(f"========================================")
    print(f"Training on region: {region}")
    print(f"========================================")
    
    # Load fresh model for each region
    model = load_model()
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    trainable_percentage = 100 * trainable_params / all_params
    print(f"Trainable params: {trainable_params:,} || All params: {all_params:,} || Trainable %: {trainable_percentage:.4f}")
    
    # Load dataset for this region
    dataset = load_dataset("nDNA/WikiCulture", region, split="train")
    ds = dataset.map(format_data, remove_columns=['page_title', 'qitem', 'bucket_geo', 'wiki_id', 'text'])
    
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=ds,
        eval_dataset=None,
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=16,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            num_train_epochs = 1,               # Number of full dataset passes. For shorter training, use `max_steps` instead (this case)
            # max_steps=1,
            learning_rate=2e-4,
            logging_steps=1,
            max_length=1024,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=6969,
            report_to="none",
        ),
    )
    
    trainer.train()
    trainer.save_model(f"/root/ndna/output/sft_checkpoints/{model_id.replace('/', '_')}/{region}")
    
    # Free memory
    del model
    del trainer
    torch.cuda.empty_cache()
    
    print(f"Completed: {region}\n")

if __name__ == "__main__":
    for region in datasets_regions:
        train_on_region(region)
    
    print("All regions completed!")