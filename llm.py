import torch
import argparse
import re
from torch.nn.functional import softmax
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

login(token = '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')
parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument("--llm", default="llama70b-nemo")
args = parser.parse_args()

if args.llm == "gemma9b":
    model_id = "google/gemma-2-9b-it"
elif args.llm == "gemma27b":
    model_id = "google/gemma-2-27b-it"
elif args.llm == "llama70b":
    model_id = "meta-llama/Meta-Llama-3.1-70B-Instruct"
elif args.llm == "llama70b-nemo":
    model_id = "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF"

# Initialize the Flask app
app = Flask(__name__)

# Load the model and tokenizer

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16)

# Define get_response function
def get_response(llm, prompt, label_keys):
    tokenizer, model = llm["tokenizer"], llm["model"]
    input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**input_ids, max_new_tokens=2048, pad_token_id=tokenizer.eos_token_id,
                             output_scores=True, return_dict_in_generate=True)
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

    # # Get the top 5 tokens and their probabilities, excluding the predicted token
    # top5_probs, top5_token_ids = torch.topk(prob_dist, k=6)  # Get top 6 to account for excluding predicted token
    # top5_tokens = []
    # top5_probs_list = []

    # for token_id, prob in zip(top5_token_ids[0].tolist(), top5_probs[0].tolist()):
    #     token_str = tokenizer.decode([token_id]).strip()
    #     if token_str != predicted_token:
    #         top5_tokens.append(token_str)
    #         top5_probs_list.append(prob)
    #     if len(top5_tokens) == 5:
    #         break

    # Build the probability distribution
    probability_distribution = {
        label: prob_dist[0, tokenizer.convert_tokens_to_ids(label)].item()
        for label in label_keys
    }
    
    # # Get probabilities for other tokens in label_keys (excluding predicted token)
    # for label in label_keys:
    #     if label != predicted_token:
    #         label_id = tokenizer.convert_tokens_to_ids(label)
    #         label_prob = prob_dist[0, label_id].item()
    #         probability_distribution['other_tokens'].append((label, label_prob))
    
    # print(response_text)
    # print(predicted_token)
    # print(probability_distribution)
    
    return response_text, probability_distribution


    # for i, (score, token_id) in enumerate(zip(outputs.scores, generated_token_ids)):
    #     # Apply softmax to get probability distribution for the current step
    #     prob_dist = softmax(score, dim=-1)
    #     # Extract the probability of the generated token
    #     token_prob = prob_dist[0, token_id].item()
    #     probabilities.append(token_prob)
    #     # Get the token string
    #     token_str = tokenizer.decode([token_id])
    #     tokens.append(token_str)
    #     # Print the token and its probability
    #     print(f"Token: '{token_str}', Probability: {token_prob}")
    
    # # Calculate probabilities for each generated token in the response
    # probabilities = []
    # for i, score in enumerate(outputs.scores):
    #     # Apply softmax to get probability distribution for the current step
    #     prob_dist = softmax(score, dim=-1)
        
    #     # Get the ID of the generated token at this step
    #     token_id = outputs.sequences[:, input_ids['input_ids'].shape[1] + i]  # Offset by prompt length
        
    #     # Extract the probability of the generated token
    #     token_prob = prob_dist[0, token_id].item()
    #     probabilities.append(token_prob)
    
    # return response_text, probabilities

# Create the LLM instance
llm = {"tokenizer": tokenizer, "model": model}

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    prompt = data.get('prompt', '')
    label_keys = data.get('label_keys', [])
    response_text, probabilities = get_response(llm, prompt, label_keys)
    return jsonify({'response_text': response_text, 'probabilities': probabilities})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
