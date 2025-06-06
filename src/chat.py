import requests
import math
from openai import OpenAI, types

client = OpenAI(
    api_key="ADD_API_KEY_HERE",
)

def chat(message: str, label_keys, seed: int, model: str ="Qwen/Qwen2.5-14B", port: str = "8000", ip: str = "localhost", is_local_client: bool | int = True):
    if is_local_client:
        url = f"http://{ip}:{port}/v1/completions"
        headers = {"Content-Type": "application/json"}
        data = {
            "model": model,
            "prompt": message,
            "temperature": 1.0,
            "max_tokens": 5,
            "logprobs": 10,
            "seed": seed
        }

        response: types.Completion = requests.post(url, headers=headers, json=data).json()
        
        text_output = response["choices"][0]["text"]
        
        logprobs_list = response["choices"][0].get("logprobs", {}).get("top_logprobs", [])
        tokens = response["choices"][0].get("logprobs", {}).get("tokens", [])
    else:
        response: types.Completion = client.completions.create(
            model=model,
            prompt=message,
            max_tokens=5,
            temperature=1.0,
            logprobs=10,
            seed=seed,
        )
        
        text_output = response.choices[0].text
        # print("\n### Full Output ###\n" + text_output)
        
        logprobs_list = response.choices[0].logprobs.top_logprobs
        tokens = response.choices[0].logprobs.tokens
    
    label_logprobs = {}
    for i, token in enumerate(tokens):
        stripped_token = token.strip()
        if stripped_token in label_keys and i < len(logprobs_list):
            full_logprobs = logprobs_list[i]
            label_logprobs = {token_option: logprob for token_option, logprob in full_logprobs.items() if token_option.strip() in label_keys}
            break  # Stop after finding the first valid label
    
    # Normalization
    exp_probs = {token: math.exp(logprob) for token, logprob in label_logprobs.items()}
    total_prob = sum(exp_probs.values())
    normalized_probs = {token: (prob / total_prob) for token, prob in exp_probs.items()} if total_prob > 0 else {}

    # print("\n### Normalized Probabilities ###\n" + str(normalized_probs))

    return text_output, normalized_probs

def chat_response_only(message: str, seed: int, max_tokens: int=10, temperature: float=1.0, model: str="Qwen/Qwen2.5-14B", port: str="8000", ip: str="localhost"):
    url = f"http://{ip}:{port}/v1/completions"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model,
        "prompt": message,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "logprobs": 10,
        "seed": seed
    }

    response = requests.post(url, headers=headers, json=data).json()
    text_output = response["choices"][0]["text"]
    
    return text_output

def chat_tabular(message: str, label_keys, seed: int):
    url = "http://localhost:8000/v1/completions"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "Qwen/Qwen2.5-14B",
        "prompt": message,
        "temperature": 1.0,
        "max_tokens": 5,
        "logprobs": 10,
        "seed": seed
    }

    response = requests.post(url, headers=headers, json=data).json()
    text_output = response["choices"][0]["text"]

    logprobs_list = response["choices"][0].get("logprobs", {}).get("top_logprobs", [])
    tokens = response["choices"][0].get("logprobs", {}).get("tokens", [])
    
    label_logprobs = {}
    full_logprobs = logprobs_list[0]
    
    stripped_logprob_keys = [token.strip() for token in full_logprobs.keys()]
    
    # check label_keys in stripped_logprob_keys
    if all(label_key in stripped_logprob_keys for label_key in label_keys):
        label_logprobs = {token_option: logprob for token_option, logprob in full_logprobs.items() if token_option.strip() in label_keys}
    
    # Normalization
    exp_probs = {token: math.exp(logprob) for token, logprob in label_logprobs.items()}
    total_prob = sum(exp_probs.values())
    normalized_probs = {token: (prob / total_prob) for token, prob in exp_probs.items()} if total_prob > 0 else {}

    return text_output, normalized_probs


def chat_openai_logprobs(message: str, label_keys: list, seed: int, model: str = "gpt-4o-mini-2024-07-18"):
    response = client.completions.create(
        model=model,
        prompt=message,
        max_tokens=5,
        temperature=1.0,
        logprobs=10,
        seed=seed,
    )
    
    text_output = response.choices[0].text
    
    # Extract token-level logprobs
    logprobs_list = response.choices[0].logprobs.top_logprobs
    tokens = response.choices[0].logprobs.tokens
    
    label_logprobs = {}
    for i, token in enumerate(tokens):
        stripped_token = token.strip()
        if stripped_token in label_keys and i < len(logprobs_list):
            full_logprobs = logprobs_list[i]
            label_logprobs = {
                tok: lp for tok, lp in full_logprobs.items() if tok.strip() in label_keys
            }
            break
        
    # Normalize
    exp_probs = {tok.strip(): math.exp(lp) for tok, lp in label_logprobs.items()}
    total_prob = sum(exp_probs.values())
    normalized_probs = {
        tok: prob / total_prob for tok, prob in exp_probs.items()
    } if total_prob > 0 else {}
    
    return text_output, normalized_probs