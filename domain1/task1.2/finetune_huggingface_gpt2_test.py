from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftConfig, PeftModel

# base_model_name = "distilgpt2"
adapter_path = "./outputs"

peft_config = PeftConfig.from_pretrained(adapter_path)
tokenizer  = AutoTokenizer.from_pretrained( peft_config.base_model_name_or_path )
base_model = AutoModelForCausalLM.from_pretrained( peft_config.base_model_name_or_path )
lora_model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=False)
lora_model.eval()

prompt = "Question: What is inflation?\nAnswer:"
prompt = "Question: What is disinflation?\nAnswer:"
prompt = "Question: What is options in finance?\nAnswer:"
prompt = "Question: What is a savings plan?\nAnswer:"
prompt = "What is the most expensive stock of all time?\nAnswer:"

inputs = tokenizer(prompt, return_tensors="pt")

outputs = lora_model.generate(
    **inputs,
    max_new_tokens=128,
    num_beams=1, 
    do_sample=True,
    temperature=0.2, 
    top_p=0.9, 
    repetition_penalty=2.5,  # 1.0 means no penalty.
    no_repeat_ngram_size=1,  # all ngrams of the `size` can only occur once.
    diversity_penalty=0.0,   # `diversity_penalty` is only effective if `group beam search` is enabled.
    length_penalty=1.0,      # `length_penalty` > 0.0 promotes longer sequences, while `length_penalty` < 0.0 encourages shorter sequences.
)

print( tokenizer.decode(outputs[0], skip_special_tokens=True) )
