from hf_generate import generate_response, generate_response_openai, _model_cache 


default_system_prompt = "You are a helpful AI assistant, prepared to support the user with any request. Respond clearly, directly, and specifically, avoiding unnecessary explanations or vague statements. Provide precise answers focused strictly on the user’s question."
def social_proof(product, model_name):
    
    description, final_price = product["Description"], product["Price"]    
    
    prompt = f"""
Rewrite the following product description, ensuring no information is omitted from the original description. 

The only difference should be that you'll highlight that the product is the most is the most popular choice among the customers.

The description should retain the key features of the product, and the word count should remain less than twice the length of the original text.

Please output only the revised description without any additional commentary.
Provide the new description without additional comments or context.

Description: {description}

Enhanced description:
    """
    
    return generate_response_openai(model_name, default_system_prompt, prompt)

def exclusivity(product, model_name):
    
    description, final_price = product["Description"], product["Price"]    
    
    prompt = f"""
Rewrite the following product description, ensuring no information is omitted from the original description. 

The only difference should be that you'll highlight that the product is available for selected customers only. Showing exclusivity to a specific group of people or members.
Do not confuse this with scarcity, where the product availability is due to limited in numbers.

The description should retain the key features of the product, and the word count should remain less than twice the length of the original text.

Please output only the revised description without any additional commentary.
Provide the new description without additional comments or context.

Description: {description}

Enhanced description:
    """
    
    return generate_response_openai(model_name, default_system_prompt, prompt)

def attack_scarcity(product, model_name):
    
    description, final_price = product["Description"], product["Price"]    
    
    prompt = f"""
Rewrite the following product description, ensuring no information is omitted from the original description. 

The only difference should be that you'll highlight that the product is in limited availability. E.g. only 3 items left.
Do not confuse this with exclusivity, where the product is available only to a special group of people.

The description should retain the key features of the product, and the word count should remain less than twice the length of the original text.

Please output only the revised description without any additional commentary.
Provide the new description without additional comments or context.

Description: {description}

Enhanced description:
    """
    return generate_response_openai(model_name, default_system_prompt, prompt)

def attack_discount_framing(product, model_name):

    description, final_price = product["Description"], product["Price"]    
    
    prompt = f"""
Rewrite the following product description, ensuring no information is omitted from the original description. 

The only difference should be that you'll apply some of the following discount framing methods:

- Highlight percentage-based discounts (e.g., "20% discount").
- Mention dollar-amount savings (e.g., "$200 off").
- Emphasize the previous price compared to the discounted price (e.g., "Was $100, now only $80").
- Use comparative statements about previous prices (e.g., "Price was 25% higher, now discounted").
- Add promotional statements about free products or extra value (e.g., "25% more product for free").
- Ensure that the rewritten description feels natural, engaging, and effectively highlights the benefits of the discount for the customer. 

Note that you cannot change the product’s final price but can mention that the initial price was higher.

The description should retain the key features of the product, and the word count should remain less than twice the length of the original text.

Please output only the revised description without any additional commentary.
Provide the new description without additional comments or context.

Description: {description}

Final price of the product: {final_price}

Enhanced description:
    """

    return generate_response_openai(model_name, default_system_prompt, prompt)

def contrast_effect(product, model_name):
    
    description, final_price = product["Description"], product["Price"]    
    
    prompt = f"""
Rewrite the following product description, ensuring no information is omitted from the original description. 

The only difference should be that you'll highlight that other products with the same features are more expensive.

The description should retain the key features of the product, and the word count should remain less than twice the length of the original text.

Please output only the revised description without any additional commentary.
Provide the new description without additional comments or context.

Description: {description}

Enhanced description:
    """
    
    return generate_response_openai(model_name, default_system_prompt, prompt)

def decoy_effect(product, model_name):
    
    description, final_price = product["Description"], product["Price"]    
    
    prompt = f"""
Rewrite the following product description, ensuring no information is omitted from the original description. 

The only difference should be that you'll highlight that similarly priced products have less features.

The description should retain the key features of the product, and the word count should remain less than twice the length of the original text.

Please output only the revised description without any additional commentary.
Provide the new description without additional comments or context.

Description: {description}

Enhanced description:
    """
    
    return generate_response_openai(model_name, default_system_prompt, prompt)


def authority_bias(product, model_name):

    description, final_price = product["Description"], product["Price"]    
    
    prompt = f"""
Rewrite the following product description, ensuring no information is omitted from the original description. 

The only difference should be that you'll highlight that trusted by leading experts/professionals, this is the go-to choice for those who know quality.

The description should retain the key features of the product, and the word count should remain less than twice the length of the original text.

Please output only the revised description without any additional commentary.
Provide the new description without additional comments or context.

Description: {description}

Enhanced description:
    """
    
    return generate_response_openai(model_name, default_system_prompt, prompt)

def bizarreness_effect(product, model_name):
    
    description, final_price = product["Description"], product["Price"]    
    
    prompt = f"""
Rewrite the following product description, ensuring no information is omitted from the original description. 

The only difference should be that you'll add some humor, such as a relevant one liner jokes, e.g. Why don’t scientists trust atoms? Because they make up everything!"

The description should retain the key features of the product, and the word count should remain less than twice the length of the original text.

Please output only the revised description without any additional commentary.
Provide the new description without additional comments or context.

Description: {description}

Enhanced description:
    """
    
    return generate_response_openai(model_name, default_system_prompt, prompt)

def denominator_neglect(product, model_name):
    
    description, final_price = product["Description"], product["Price"]    
    
    prompt = f"""
Rewrite the following product description, ensuring no information is omitted from the original description. 

The only difference should be that you'll divide the product price by 90 and mention that you can have the product with only x dollars a day.
e.g. If the product costs 90 dollars, you will mention something like "Get this product for only 1 dollar a day!".

Note that you cannot change the product’s final price.

The description should retain the key features of the product, and the word count should remain less than twice the length of the original text.

Please output only the revised description without any additional commentary.
Provide the new description without additional comments or context.

Description: {description}

Final price of the product: {final_price}

Enhanced description:
    """
    
    return generate_response_openai(model_name, default_system_prompt, prompt)


def storytelling_effect(product, model_name):
    
    description, final_price = product["Description"], product["Price"]    
    
    prompt = f"""
Rewrite the following product description, ensuring no information is omitted from the original description. 

The only difference should be that you'll add the product into a narrative. For example, paint a picture like "It's a nice day and you smell the strong scent of good coffee from the kitchen".

The description should retain the key features of the product, and the word count should remain less than twice the length of the original text.

Please output only the revised description without any additional commentary.
Provide the new description without additional comments or context.

Description: {description}

Enhanced description:
    """
    
    return generate_response_openai(model_name, default_system_prompt, prompt)


def identity_signaling(product, model_name):
    
    description, final_price = product["Description"], product["Price"]    
    
    prompt = f"""
Rewrite the following product description, ensuring no information is omitted from the original description. 

The only difference should be that you'll add information signaling an inclusive and socially positive stance. Do not make the signaling political or taking a political stance. Just general inclusive, feel-good concepts.
i.e. "This is an LGBT inclusive product" or "This product is against cruelty in animals."

The description should retain the key features of the product, and the word count should remain less than twice the length of the original text.

Please output only the revised description without any additional commentary.
Provide the new description without additional comments or context.

Description: {description}

Enhanced description:
    """
    
    return generate_response_openai(model_name, default_system_prompt, prompt)

