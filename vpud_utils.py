import math
import re

def calculate_entropy(probs):
    # Calculate entropy using all probabilities in the dictionary
    entropy = 0.0
    for p in probs.values():
        if p > 0:
            entropy -= p * math.log2(p)
    return round(entropy, 5)

def extract_label(predicted_output: str):
    match = re.search(r'<output>\s*(.*?)\s*</output>', predicted_output, re.DOTALL | re.IGNORECASE)
    if match:
        extracted_label = match.group(1).strip()
    else:
        print("Could not find output tags in the response.")
        raise ValueError("Invalid response format.")
    return extracted_label