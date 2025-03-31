import requests
import math


#prompt = sys.argv[1]
import numpy as np
import csv

with open('prompt_neg.txt', 'r') as file:
    prompt = file.read()

API_URL = "http://localhost:8000/v1/completions"  # Correct vLLM endpoint

# Request data
data = {
    "prompt": prompt,
    "max_tokens": 5,  # Only get the next token
}

save_freq = 10
samples = []
num_samples = 10000
stop_token = "<"
# Send request
for s in range(num_samples):
    
    generated_text = ""
    valid_number = True
    while stop_token not in generated_text:
        

        response = requests.post(API_URL, json=data).json()

        data = {
            "prompt": prompt + generated_text,  # Continue from previous output
            "max_tokens": 1,  # Generate a few tokens at a time
            "logprobs": 20
        }

        response = requests.post(API_URL, json=data).json()
        
        if "choices" in response and len(response["choices"]) > 0:
            new_text = response["choices"][0]["text"]
            generated_text += new_text  # Append new text
        else:
            print("Error: No response from model")
            break
        
        if len(generated_text)>0:
            try: float(generated_text)
            except:
                
                if generated_text not in [".", "-", "-."]:
                    valid_number=False
                    print("not valid, " + generated_text)
                    break
            
        
    if valid_number:
        generated_text = generated_text.rsplit(stop_token)[0]
        samples.append(generated_text)
        # Save samples to a CSV file
        if s%10 == 0:    
            print(f"Sample {s} of {num_samples}")
            with open('samples.csv', 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                #writer.writerow(["Sample"])  # Header
                for sample in samples:
                    writer.writerow([sample])