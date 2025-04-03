import numpy as np
import requests
import math

def merge_logprobs(d1, d2):
    merged = {key: np.logaddexp(d1.get(key, -np.inf), d2.get(key, -np.inf)) for key in set(d1) | set(d2)}
    return merged

def get_pdf(
        prompt: str,
        max_api_calls: int,
        max_dp: int,
        min_y: float,
        max_y: float,
        api_url="http://localhost:8000/v1/completions",
        branching_number=15,
        stop_token = "<",
        threshold_logprob = -np.inf,
        include_unfinished=True):
    logprob_dict = get_prob_bins(
        prompt,
        max_api_calls,
        api_url,
        branching_number=branching_number,
        stop_token=stop_token,
        threshold_logprob=threshold_logprob,
        include_unfinished=include_unfinished)
    pdf = get_pdf_from_bins(logprob_dict,
                      max_dp,
                      min_y,
                      max_y,
                      ):
    return pdf


def get_prob_bins(
        prompt: str,
        max_api_calls: int,
        api_url="http://localhost:8000/v1/completions",
        branching_number=15,
        stop_token = "<",
        threshold_logprob = -np.inf,
        include_unfinished=True):
    
    #braching number and max_api_calls will be replaced by probability threshold
    original_prompt = prompt

    data = {
    "prompt": original_prompt,
    "max_tokens": 1,
    "logprobs": branching_number
    }
    
    branches = {}
    finished_branches = []
    avail_branches = {}
    next_logprob = 0
    calls = 0
    while calls <= max_api_calls and next_logprob > threshold_logprob and len(avail_branches) > 0:
        calls +=1
        logprob = next_logprob
        next_digits = max(avail_branches, key=branches.get)

        data = {
            "prompt": original_prompt+next_digits,
            "max_tokens": 1,
            "logprobs": branching_number
        }

        response = requests.post(api_url, json=data).json()
        # Convert logprobs to probabilities
        top_tokens = response["choices"][0]["logprobs"]["top_logprobs"][0]  # First token's probabilities
        filtered_tokens = {
            token: llm_logprob + logprob  # Convert log probability to probability and combine with previous probability
            for token, llm_logprob in top_tokens.items()
            if logprob + llm_logprob > threshold_logprob  # Only keep tokens above threshold log probability
        }
        new_branches = {next_digits+t:filtered_tokens[t] for t in filtered_tokens}        
        
        
        for b in new_branches:
            if stop_token in b:
                finished_branches.append(b)
        
        del branches[next_digits]
        branches = merge_logprobs(branches, new_branches)
        avail_branches = {key : branches[key] for key in branches if key not in finished_branches}
        next_logprob = max(avail_branches.values())

    #process the finished branches
    logprob_dict = {}
    for b in branches:
        if stop_token in b or "." in b or include_unfinished:
            bin_label = b
            if stop_token in bin_label:       #if stop token is there, remove it
                bin_label = bin_label.split(stop_token)[0]
                print("stop token in b", bin_label)
            bin_label = bin_label.strip()
            bin_label = bin_label.lstrip("0")
            if len(bin_label) == 0:
                bin_label = "0"
            if bin_label[0] == ".":
                bin_label = "0" + bin_label
            if bin_label[-1] == ".": #if decimal point is at the end, remove it
                bin_label = bin_label[:-1]
            try:
                assert(abs(float(bin_label)))<np.inf
                assert(float(bin_label) == float(bin_label)) #check if it is a valid number, not Nan
                if bin_label not in logprob_dict:
                    logprob_dict[bin_label] = branches[b]
                else: #if number is already there, add probabilities
                    logprob_dict[bin_label] = np.logaddexp(logprob_dict[bin_label], branches[b])
            except: #if it is not a valid number
                print("Invalid number", bin_label)
    
    return logprob_dict#, finished_branches

def get_pdf_from_bins(logprob_dict,
                     # finished_branches,
                      max_dp: int,
                      min_y: float,
                      max_y: float,
                      ):
    probs = {key: math.exp(float(value)) for key, value in logprob_dict.items()}

    max_decimals = max(len(key.split(".")[1]) if "." in key else 0 for key in probs.keys()) #max decimal in keys
    y_vals = np.linspace(min_y, max_y, int((min_y-max_y)/max_dp))
    pdf = []
    for x in y_vals:
        height = 0
        x_str = str(x)
        if "." not in x_str:
            bin = x_str
            if bin in probs:
                height += probs[bin]
        else:
            while len(x_str.split(".")[1])<max_decimals:
                x_str += "0"
            bin = str(int(x))
            if bin in probs:
                height += probs[bin]
            for i in range(max_decimals):
                bin = x_str.split(".")[0]+"."+x_str.split(".")[1][:i+1]
                if bin in probs:
                    height += probs[bin]*10**(i+1)
        pdf.append([x, height])
    
    return np.array(pdf)