from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json 

def products_to_string(d):
    text = ""
    for row in d:
        text += str(row) + "\n"
    return text

def create_prompt(list_of_products, user_msg):
    USER_PROMPT = f"""
Product Information ↓
Products:

{products_to_string(list_of_products)}

{user_msg}
    """
    return USER_PROMPT


def get_user_msg(catalog, user_msg_type):
    if catalog == "coffee_machines":
        filename = "data/coffee_machines.jsonl"
        if user_msg_type == "abstract":
            user_msg = "I am looking for a coffee machine. Can I get some recommendations?"
        elif user_msg_type == "specific":
            user_msg = "I am looking for an affordable coffee machine. Can I get some recommendations?"


    elif catalog == "books":
        filename = "data/books.jsonl"
        if user_msg_type == "abstract":
            user_msg = "I am looking for a book. Can I get some recommendations?"
        elif user_msg_type == "specific":
            user_msg = "I am looking for a good adventure novel. Can I get some recommendations?"

    elif catalog == "cameras":
        filename = "data/cameras.jsonl"
        if user_msg_type == "abstract":
            user_msg = "I am looking for a camera. Can I get some recommendations?"
        elif user_msg_type == "specific":
            user_msg = "I am looking for a high resolution camera. Can I get some recommendations?"
            
    elif catalog == "laptops":
        filename = "data/amazon_filtered_by_rating/laptops.jsonl"
        if user_msg_type == "abstract":
            user_msg = "I am looking for a laptop. Can I get some recommendations?"
        elif user_msg_type == "specific":
            user_msg = "I am looking for an affordable laptop. Can I get some recommendations?" #alternatives: powerful, durable, compact
            
    elif catalog == "home_office_chairs":
        filename = "data/amazon_filtered_by_rating/home_office_chairs.jsonl"
        if user_msg_type == "abstract":
            user_msg = "I am looking for a home office chair. Can I get some recommendations?"
        elif user_msg_type == "specific":
            user_msg = "I am looking for an affordable home office chair. Can I get some recommendations?" #alternatives: ergonomic, durable
            
    elif catalog == "chew_toys":
        filename = "data/amazon_filtered_by_rating/chew_toys.jsonl"
        if user_msg_type == "abstract":
            user_msg = "I am looking for a chew toy. Can I get some recommendations?"
        elif user_msg_type == "specific":
            user_msg = "I am looking for an affordable chew toy. Can I get some recommendations?" #alternatives: durable

    elif catalog == "binoculars":
        filename = "data/binoculars.jsonl"
        if user_msg_type == "abstract":
            user_msg = "I am looking for binoculars. Can I get some recommendations?"
        elif user_msg_type == "specific":
            user_msg = "I am looking for affordable binoculars. Can I get some recommendations?" #alternatives: durable

    else:
        raise ValueError("Invalid catalog.")
    
    return user_msg, filename


def read_products(filename):
    with open(filename, 'r') as json_file:
        json_list = list(json_file)

    results = []
    for json_str in json_list:
        results.append(json.loads(json_str))
    return results



# Global cache for models and tokenizers
_model_cache = {}

def generate_response(model_name, system_prompt, user_prompt, max_new_tokens=None, temperature=0.0):
    """
    Downloads and runs a model from Hugging Face.
    Uses caching to avoid re-downloading the model on subsequent calls.
    
    Args:
        model_name (str): The model name on HF (e.g., "Qwen/Qwen2.5-0.5B-Instruct")
        system_prompt (str): The system prompt
        user_prompt (str): The user prompt
        max_new_tokens (int): Maximum number of tokens for the response (None = no limit)
        temperature (float): Temperature for generation (0.0 = deterministic)
    
    Returns:
        str: The model's response
    """
    
    # Check if model is already in cache
    if model_name not in _model_cache:
        print(f"Loading model: {model_name}...")
        
        # Download tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # Store in cache
        _model_cache[model_name] = {
            'tokenizer': tokenizer,
            'model': model
        }
        # print(f"Model loaded and cached.")


    # Get model and tokenizer from cache
    tokenizer = _model_cache[model_name]['tokenizer']
    model = _model_cache[model_name]['model']
    
    # Create chat format (for instruction-tuned models)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    
    # Tokenization
    model_inputs = tokenizer([text], return_tensors="pt")
    if torch.cuda.is_available():
        model_inputs = model_inputs.to("cuda")
    
    
    # Prepare generation kwargs
    gen_kwargs = {
        'input_ids': model_inputs.input_ids,
        'temperature': temperature,
        'do_sample': True if temperature > 0 else False,
        'pad_token_id': tokenizer.eos_token_id,
        'max_new_tokens': 100000
    }
    
    # Only add max_new_tokens if specified
    if max_new_tokens is not None:
        gen_kwargs['max_new_tokens'] = max_new_tokens
    
    generated_ids = model.generate(**gen_kwargs)
    
    # Decode response
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return response

SYSTEM_PROMPT = "A chat between a human and an artificial intelligence assistant. The assistant provides a numbered list of product recommendations ranked based on the user’s request."

user_msg, filename = get_user_msg("binoculars", "abstract")

list_of_products = read_products(filename)

user_prompt = create_prompt(list_of_products, user_msg)

resp = generate_response("Qwen/Qwen3-0.6B",
                 system_prompt = SYSTEM_PROMPT,
                 user_prompt = user_prompt)

print (f"Response: {resp}")
