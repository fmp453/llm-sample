# reference : https://www.kaggle.com/code/changyeop/how-to-fine-tune-gpt-2-for-beginners

from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import Trainer, TrainingArguments

from datasets import load_dataset

def fine_tune():

    # load model and tokenizer
    model_name = "openai-community/gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # make dataset
    # file_path = ""
    dataset = load_dataset("lhoestq/demo1")

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False # masked language modelingをするか
    )

    training_args = TrainingArguments(
        output_dir="out",
        do_train=True,
        per_device_train_batch_size=8,
        num_train_epochs=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset
    )

    trainer.train()
    trainer.save_model()

def main():
    fine_tune()

if __name__ == "__main__":
    main()
