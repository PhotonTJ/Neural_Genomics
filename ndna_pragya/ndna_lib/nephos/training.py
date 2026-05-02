# Model training and fine-tuning logic for Nephos
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

def train_nephos_model(
	tokenized_data: list,
	base_model: str = "meta-llama/Llama-3.2-1B-Instruct",
	output_dir: str = "./nephos-v2-llama",
	lora_r: int = 16,
	lora_alpha: int = 32,
	lora_dropout: float = 0.05,
	batch_size: int = 2,
	grad_accum: int = 8,
	max_steps: int = 200,
	learning_rate: float = 2e-4,
	fp16: bool = True,
	logging_steps: int = 10,
	save_total_limit: int = 2,
	hf_token: str = None
) -> tuple:
	"""
	Sets up and runs model training using LoRA/PEFT and HuggingFace Trainer.
	Returns the trained model and tokenizer.
	"""
	tokenizer = AutoTokenizer.from_pretrained(base_model, token=hf_token)
	tokenizer.pad_token = tokenizer.eos_token
	model = AutoModelForCausalLM.from_pretrained(base_model, token=hf_token, device_map="auto")
	model.resize_token_embeddings(len(tokenizer))

	peft_config = LoraConfig(
		r=lora_r,
		lora_alpha=lora_alpha,
		target_modules=["q_proj", "v_proj"],
		lora_dropout=lora_dropout,
		bias="none",
		task_type="CAUSAL_LM",
	)
	model = get_peft_model(model, peft_config)

	# Convert list of dicts to HuggingFace Dataset
	train_dataset = Dataset.from_list(tokenized_data)

	training_args = TrainingArguments(
		per_device_train_batch_size=batch_size,
		gradient_accumulation_steps=grad_accum,
		warmup_steps=10,
		max_steps=max_steps,
		learning_rate=learning_rate,
		logging_steps=logging_steps,
		output_dir=output_dir,
		save_total_limit=save_total_limit,
		fp16=fp16,
		report_to="none",
	)

	trainer = Trainer(
		model=model,
		args=training_args,
		train_dataset=train_dataset,
		tokenizer=tokenizer,
	)

	trainer.train()
	model.save_pretrained(output_dir)
	tokenizer.save_pretrained(output_dir)
	return model, tokenizer
