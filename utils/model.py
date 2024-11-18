from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer, PreTrainedModel


def get_model(
        model_name: str,
        token=""
) -> AutoModelForCausalLM:
    """
    Get the model from the model name

    Args:
        model_name (str): Name of the model
        token (str): Token

    Returns:
        AutoModelForCausalLM: Model object
    """
    model = AutoModelForCausalLM.from_pretrained(model_name, token=token)
    return model


def get_model_and_tokenizer(
        model_name: str,
        model_max_length: int,
        token=""
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Get the model from the model name

    Args:
        model_name (str): Name of the model
        model_max_length (int): Maximum length of the model
        token (str): Token

    Returns:
        Tuple[AutoModelForCausalLM, AutoTokenizer]: Model object and Tokenizer object
    """
    model = AutoModelForCausalLM.from_pretrained(model_name, token=token)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        model_max_length=model_max_length,
        padding_side="right",
        use_fast=False,
        token=token
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer

