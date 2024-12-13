import torch
import argparse
import re
from torch.nn.functional import softmax
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import set_seed
from huggingface_hub import login

parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument("--llm", default="llama70b-nemo")
args = parser.parse_args()

# Initialize the Flask app
app = Flask(__name__)

login(token = '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')

if args.llm == "gemma9b":
    model_id = "google/gemma-2-9b-it"
elif args.llm == "gemma27b":
    model_id = "google/gemma-2-27b-it"
elif args.llm == "llama70b":
    model_id = "meta-llama/Meta-Llama-3.1-70B-Instruct"
elif args.llm == "llama70b-nemo":
    model_id = "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF"

# Load the model and tokenizer

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16)

# Create the LLM instance
llm = {"tokenizer": tokenizer, "model": model}

# Define get_response function
def get_response(llm, prompt, label_keys, seed):
    
    # print("seed", seed)
    set_seed(seed)
    
    
    tokenizer, model = llm["tokenizer"], llm["model"]
    input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**input_ids, max_new_tokens=2048, pad_token_id=tokenizer.eos_token_id,
                             output_scores=True, return_dict_in_generate=True, do_sample = True, temperature = 0.5)
    # outputs = model.generate(**input_ids, max_new_tokens=2048, pad_token_id=tokenizer.eos_token_id,
                            #  output_scores=True, return_dict_in_generate=True, do_sample = True, top_p=0.9, top_k = 50)
    gen_text = tokenizer.decode(outputs.sequences[0])
    
    # Find the starting point of the prompt in the generated text
    start_pos = gen_text.find(prompt)
    if start_pos == -1:
        return "Prompt not found in the generated text."

    # Extract the response starting from the prompt
    response_text = gen_text[start_pos + len(prompt):].strip()

    # Find the end position of <end_of_turn> if it exists -- Gemma
    end_pos = response_text.find("<end_of_turn>")
    if end_pos != -1:
        response_text = response_text[:end_pos].strip()

    # Find the end position of <end_of_turn> if it exists -- Llama
    end_pos = response_text.find("<|eot_id|>")
    if end_pos != -1:
        response_text = response_text[:end_pos].strip()
    
    # Process end-of-turn tags for different models
    for end_tag in ["<end_of_turn>", "<|eot_id|>"]:
        end_pos = response_text.find(end_tag)
        if end_pos != -1:
            response_text = response_text[:end_pos].strip()

    # Extract the predicted token within <output> </output> tags
    match = re.search(r'<output>\s*(.*?)\s*</output>', response_text)
    if not match:
        print("Prediction not found in expected format.")
        exit()

    predicted_token = match.group(1).strip()
    if predicted_token not in label_keys:
        print(f"Predicted token '{predicted_token}' not in label_keys.")
        exit()

    # Now find the position where the predicted token was generated
    # Tokenize the predicted token to get its token ids
    predicted_token_ids = tokenizer(predicted_token, add_special_tokens=False)['input_ids']

    # Get the generated token ids (excluding the input prompt)
    generated_token_ids = outputs.sequences[0][input_ids['input_ids'].shape[1]:].tolist()
    
    # Find the index where the predicted token starts in generated_token_ids
    def find_sublist(sublist, main_list):
        for i in range(len(main_list) - len(sublist) + 1):
            if main_list[i:i+len(sublist)] == sublist:
                return i
        return -1

    start_idx = find_sublist(predicted_token_ids, generated_token_ids)
    if start_idx == -1:
        return "Predicted token ids not found in generated token ids.", None, None

    # At the position where the predicted token starts, get the probability distribution
    # Get the score at that position
    score = outputs.scores[start_idx]

    # Get the probabilities
    prob_dist = softmax(score, dim=-1)

    # Build the probability distribution
    probability_distribution = {
        label: prob_dist[0, tokenizer.convert_tokens_to_ids(label)].item()
        for label in label_keys
    }

    total_prob = sum(probability_distribution.values())
    normalized_probability_distribution = {
        label: round(prob / total_prob, 5)
        for label, prob in probability_distribution.items()
    }
    
    return response_text, normalized_probability_distribution

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    prompt = data.get('prompt', '')
    label_keys = data.get('label_keys', [])
    seed = data.get('seed')
    response_text, probabilities = get_response(llm, prompt, label_keys, seed)
    return jsonify({'response_text': response_text, 'probabilities': probabilities})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
