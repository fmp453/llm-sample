
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import Trainer, TrainingArguments

def peft_tune():

    # load model and tokenizer
    model_name = "openai-community/gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    from peft import LoraConfig, get_peft_model

    peft_config = LoraConfig(
        r=4,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=4,
        lora_dropout=0,
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        per_device_train_batch_size=4, 
        gradient_accumulation_steps=4,
        warmup_steps=25, 
        max_steps=50, 
        learning_rate=2e-4,
        logging_steps=1, 
        output_dir='outputs'
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args
    )

    trainer.train()