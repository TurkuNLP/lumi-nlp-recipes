import os
import sys
import random
import torch
from data import get_datasets
from trl import (
    SFTTrainer,
    setup_chat_format
)
from transformers import (
    HfArgumentParser,
    set_seed
)
from accelerate import logging

#Relative imports
from configs import ModelArguments,DataArguments,SftArguments
from utils import get_tokenizer,get_peft_config
#from data import apply_chat_template
from datasets import load_dataset

logger = logging.get_logger(__name__)

def main(model_args, data_args, training_args):
    
    # Set seed for reproducibility
    set_seed(training_args.seed)

    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training parameters{training_args}")


    ##DATA##
    #Load from a jsonl file
    #dataset = load_dataset("json", data_files="my_file.json",split="train")
    #Load from hf hub
    dataset = load_dataset("Villekom/ultrachat-16k-fi-oai",split="train")


    ##Tokenizer##
    tokenizer = get_tokenizer(model_args,data_args)


    #Apply chat template
    #More info on these: https://huggingface.co/docs/transformers/main/en/chat_templating
    #https://huggingface.co/docs/trl/main/en/sft_trainer#add-special-tokens-for-chat-format
    with training_args.main_process_first():
        dataset = dataset.map(lambda x: {"text": tokenizer.apply_chat_template(x['messages'],tokenize=False,add_generation_prompt=False)})
    

    #Log some examples from the dataset
    #for index in random.sample(range(len(dataset["train"])), 3):
    #    logger.info(f"Sample {index} of the processed training set:\n\n{raw_datasets['train'][index]['text']}")

    #Model kwargs
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    model_kwargs = dict(
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
    )
    
    ########################
    # Initialize the Trainer
    ########################
    trainer = SFTTrainer(
        model=model_args.model_name_or_path,
        model_init_kwargs=model_kwargs,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        peft_config=get_peft_config(model_args),
        packing=data_args.packing,
        dataset_text_field=data_args.dataset_text_field,
        max_seq_length=model_args.max_seq_length,
        dataset_kwargs={
            "append_concat_token": data_args.append_concat_token,
            "add_special_tokens": data_args.add_special_tokens,
        }
    )
    
    #Wait for all the ranks to get to this point, so training does not crash/hang
    trainer.accelerator.wait_for_everyone()
    logger.debug(f"Model: {trainer.model}")
    
    ##Training loop
    logger.info("Starting training")
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    #Save model
    trainer.save_state()
    trainer.save_model()

    logger.info("Training complete")

if __name__ == "__main__":
    parser = HfArgumentParser([ModelArguments,DataArguments,SftArguments])

    model_args, data_args, training_args = parser.parse_yaml_file(os.path.abspath(sys.argv[1]),allow_extra_keys=True)
    
    main(model_args, data_args, training_args)