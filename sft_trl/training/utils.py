from transformers import PreTrainedTokenizer, AutoTokenizer
from configs import ModelArguments, DataArguments
from data import DEFAULT_CHAT_TEMPLATE
from peft import PeftConfig,LoraConfig

def get_tokenizer(
    model_args: ModelArguments, data_args: DataArguments, auto_set_chat_template: bool = True
) -> PreTrainedTokenizer:
    """Get the tokenizer for the model."""
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    # Set a reasonable default for models without max length
    # This is the case for Poro
    if tokenizer.model_max_length > 100_000:
        tokenizer.model_max_length = 2048

    if data_args.chat_template is not None:
        tokenizer.chat_template = data_args.chat_template

    return tokenizer


def get_peft_config(model_args: ModelArguments) -> PeftConfig | None:
    if model_args.use_peft is False:
        return None
    peft_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=model_args.lora_target_modules,
        modules_to_save=model_args.lora_modules_to_save,
    )

    return peft_config