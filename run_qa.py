import re
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.dataset import QADataset
from src.prompt import Prompt
from src.chat import chat_tabular
from src.utils import calculate_entropy, calculate_kl_divergence, QAUtils

# note to self: play around with num_seeds and z_samples

# Main
def main():
    global_seed = int(args.seed)

    args.id_dataset = "boolq"

    qa = QADataset(args.id_dataset)            # only dataset currently supported
    train, test, label_keys = qa.load_data()
    train_ood, test_ood = qa.generate_ood()

    # Sampling D as icl
    num_D = len(train) - 1
    df_D = train.sample(n=num_D, random_state=global_seed)
    train = train.drop(df_D.index)

    # Sample z
    df_z = train.sample(n=1, random_state=global_seed)
    train = train.drop(df_z.index)
    num_z = 20
    
    # Sampling x
    num_x = len(test)
    data_x = test.sample(n=num_x, random_state=global_seed)
    test = test.drop(data_x.index) # drop the sampled x

    prompt = Prompt(prompt_type="tabular")

    x_z_lst = []

    for i, x_row in tqdm(data_x.iterrows(), total=len(data_x), desc="Processing x perturbations"):
        x = x_row['note']
        min_Va_lst = []
        seed = 0

        data_z = QAUtils.perturb_z(data=df_D, x_row=x_row, z_samples=num_z)
        for i, row in tqdm(data_z.iterrows(), total=len(data_z), desc="Processing z perturbations"):
            seed = 0
            z = row['note']

            # Initialize dictionaries to store average probabilities
            avg_puzD_probs = {label: 0.0 for label in label_keys}
            avg_pyxuzD_probs = {f"p(y|x,u{outer_label},z,D)": {inner_label: 0.0 for inner_label in label_keys} for outer_label in label_keys}
            avg_pyxzD_probs = {label: 0.0 for label in label_keys}
            avg_pyxD_probs = {label: 0.0 for label in label_keys}

            # Extracting prompt probabilities: p(u|z,D), p(y|x,u,z,D), p(y|x,D)
            successful_seeds = 0
            successful_seeds_lst = []
            
            num_seeds = args.num_seeds  # target number of successful seeds
            while successful_seeds < num_seeds:

                # Create temporary dictionaries for this seed
                temp_avg_puzD = {label: 0.0 for label in label_keys}
                temp_avg_pyxuzD = {f"p(y|x,u{outer_label},z,D)": {inner_label: 0.0 for inner_label in label_keys} for outer_label in label_keys}
                temp_avg_pyxD = {label: 0.0 for label in label_keys}

                ## Shuffle D
                df_D_shuffled = df_D.sample(frac=1, random_state=seed).reset_index(drop=True)
                D = "\n".join(
                    [f"{row['note']} <output>{row['label']}</output>\n" for _, row in df_D_shuffled.iterrows()]
                )
                
                # p(u|z,D)
                prompt_puzD = prompt.get_puzD_prompt(z, D)
                output_puzD, puzD = chat_tabular(prompt_puzD, label_keys, seed)
                if not re.search(r'\d+</output>', output_puzD):
                    seed += 1
                    continue
                if not puzD:
                    seed += 1
                    continue
                for label, prob in puzD.items():
                    temp_avg_puzD[label] += prob
                
                # p(y|x,u,z,D)
                skip_seed = False
                dict_uz = {}
                for label_key in label_keys:
                    df_copy = df_z.copy()
                    df_copy["label"] = label_key
                    dict_uz[f"u{label_key}z"] = df_copy

                dict_uzD = {}
                for key, df_uz in dict_uz.items():
                    dict_uzD[f"{key}D"] = pd.concat([df_uz, df_D], ignore_index=True)
                    # Shuffle u,z,D
                    dict_uzD[f"{key}D"] = dict_uzD[f"{key}D"].sample(frac=1, random_state=seed).reset_index(drop=True)

                prompt_uzD = {}
                for key, df_uzD in dict_uzD.items():
                    prompt_uzD[f"{key}"] = "\n".join([f"{row['note']} <output>{row['label']}</output>\n" for _, row in df_uzD.iterrows()])

                for key, icl in prompt_uzD.items():
                    u_value = re.search(r"u(\d+)", key).group(1)  # Match 'u' followed by digits
                    prompt_pyxuzD = prompt.get_pyxuzD_prompt(x, icl)
                    output_pyxuzD, pyxuzD = chat_tabular(prompt_pyxuzD, label_keys, seed)
                    if not re.search(r'\d+</output>', output_pyxuzD):
                        skip_seed = True
                        break
                    if not pyxuzD:
                        skip_seed = True
                        break
                    for label, prob in pyxuzD.items():
                        temp_avg_pyxuzD[f"p(y|x,u{u_value},z,D)"][label] += prob
                if skip_seed:
                    seed += 1
                    continue

                # p(y|x,D)
                prompt_pyxD = prompt.get_pyxD_prompt(x, D)
                output_pyxD, pyxD = chat_tabular(prompt_pyxD, label_keys, seed)
                if not re.search(r'\d+</output>', output_pyxD):
                    seed += 1
                    continue
                if not pyxD:
                    seed += 1
                    continue
                for label, prob in pyxD.items():
                    temp_avg_pyxD[label] += prob

                # Only update the global accumulators if all outputs are valid
                for label in label_keys:
                    avg_puzD_probs[label] += temp_avg_puzD[label]
                    avg_pyxD_probs[label] += temp_avg_pyxD[label]
                    for u_label in label_keys:
                        avg_pyxuzD_probs[f"p(y|x,u{u_label},z,D)"][label] += temp_avg_pyxuzD[f"p(y|x,u{u_label},z,D)"][label]

                successful_seeds += 1
                successful_seeds_lst.append(seed)
                seed += 1

            print("Successful Seeds List:", successful_seeds_lst)

            # Average probabilities
            for label in label_keys:
                avg_puzD_probs[label] /= num_seeds
                avg_pyxD_probs[label] /= num_seeds
                for u_label in label_keys:
                    avg_pyxuzD_probs[f"p(y|x,u{u_label},z,D)"][label] /= num_seeds

            # p(y|x,z,D) via marginalization
            for label in label_keys:
                avg_pyxzD_probs[label] = sum(
                    avg_pyxuzD_probs[f"p(y|x,u{u_label},z,D)"][label] * avg_puzD_probs[u_label]
                    for u_label in avg_puzD_probs.keys()
                )

            ## Thresholding p(y|x,z,D) and p(y|x,D) via KL Divergence
            kl = calculate_kl_divergence(avg_pyxzD_probs, avg_pyxD_probs)
            data_z.at[i, 'KL'] = kl

        min_kl = data_z.sort_values('KL').head(5)
        for _, row in min_kl.iterrows():
            # Compute Va for each of the bottom 3 KL perturbations
            H_pyxuzD = {}
            for key, probs in avg_pyxuzD_probs.items():
                H_pyxuzD[f"H[{key}]"] = calculate_entropy(probs)

            E_H_pyxuzD = sum(
                H_pyxuzD[f"H[p(y|x,u{u_value},z,D)]"] * puzD[u_value] for u_value in puzD.keys()
            )

            row['Va'] = E_H_pyxuzD
            min_Va_lst.append(row)

        H_pyxD = calculate_entropy(avg_pyxD_probs)

        min_Va = min(min_Va_lst, key=lambda row: row['Va'])
        min_Va['TU'] = H_pyxD

        Ve = H_pyxD - min_Va['Va']
        min_Va['Ve'] = Ve

        pred_label = max(avg_pyxD_probs, key=avg_pyxD_probs.get)
        true_label = x_row['label']
            
        x_z = {
            'x_note': x_row['note'],
            'TU': min_Va['TU'],
            'Va': min_Va['Va'],
            'Ve': min_Va['Ve'],
            'true_label': true_label,
            'pred_label': pred_label,
        }
        x_z = pd.DataFrame([x_z])
        x_z_lst.append(x_z)

        df_plot = pd.concat(x_z_lst, ignore_index=True)
        df_plot.to_csv(f"results_tabular/OOD/df_{args.distb}_ID-{args.id_dataset}_OOD-{args.ood_dataset}_{num_D}ICL_{num_z}z.csv", index=False)

if __name__ == "__main__":
    # Argument Parser
    pd.set_option('display.max_columns', None)
    parser = argparse.ArgumentParser(description='Run VPUD')
    parser.add_argument("--seed", default=123)
    parser.add_argument("--id_dataset", default="iris")
    parser.add_argument("--ood_dataset", default="beijing")
    parser.add_argument("--num_seeds", default=5)
    parser.add_argument("--distb", default="OOD")
    args = parser.parse_args()
    main()
