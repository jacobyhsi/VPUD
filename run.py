import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.dataset import load_dataset
from src.prompt import Prompt
from src.chat import chat_tabular
from src.utils import calculate_entropy, calculate_kl_divergence, TabularUtils

# note to self: play around with num_seeds and z_samples

# Main
def main():
    # Load dataset
    data, test, label_keys = load_dataset(args.data_path)

    global_seed = int(args.seed)
    # Sampling x
    x_row = test.sample(n=1, random_state=global_seed)
    test = test.drop(x_row.index) # drop the sampled x
    x = x_row['note'].iloc[0]
    # print("x:", x)
    # x_y = x_row['label'].iloc[0]
    # print("x label:", x_y)
    # Perturb x
    # x perturbations should be in a range of values from the min of capital gains to the max of capital gains
    perturb_x = args.perturb_x
    data_x = TabularUtils.perturb_x(data, x_row, perturb_x)

    # Sampling D as icl
    num_D = 20
    df_D = data.sample(n=num_D, random_state=global_seed)
    data = data.drop(df_D.index)

    # Sample z
    df_z = data.sample(n=1, random_state=global_seed)
    data = data.drop(df_z.index)
    z = df_z['note'].iloc[0]
    # print("z:", z)
    # data_z = TabularUtils.perturb_z(data, df_z, z_samples=5) # z_samples number of pertubations per z; higher dims, more zs
    
    # perturbed x should be close to x.

    prompt = Prompt(prompt_type="tabular")
    
    x_z_lst = []
    failed_seeds = 0

    # Processing x perturbations
    for i, x_row in tqdm(data_x.iterrows(), total=len(data_x), desc="Processing x perturbations"):
        # print(f"\nProcessing x pertubation: {i}/{len(data_x)}")
        x = x_row['note']
        # Processing z Probabilities
        min_Va_lst = []

        data_z = TabularUtils.perturb_z(data, df_z, x_row, z_samples=10)

        for i, row in tqdm(data_z.iterrows(), total=len(data_z), desc="Processing z perturbations"):
            z = row['note']

            # Initialize dictionaries to store average probabilities
            avg_puzD_probs = {label: 0.0 for label in label_keys}
            avg_pyxuzD_probs = {f"p(y|x,u{outer_label},z,D)": {inner_label: 0.0 for inner_label in label_keys} for outer_label in label_keys}
            avg_pyxzD_probs = {label: 0.0 for label in label_keys}
            avg_pyxD_probs = {label: 0.0 for label in label_keys}

            # Extracting prompt probabilities: p(u|z,D), p(y|x,u,z,D), p(y|x,D)
            successful_seeds = 0
            successful_seeds_lst = []
            seed = 0

            num_seeds = args.num_seeds  # target number of successful seeds
            while successful_seeds < num_seeds:
                # print(f"\nSample: {i}; Seed: {seed + 1}")
                # print(f"Successful Seeds: {successful_seeds}/{num_seeds}")

                # Create temporary dictionaries for this seed
                temp_avg_puzD = {label: 0.0 for label in label_keys}
                temp_avg_pyxuzD = {f"p(y|x,u{outer_label},z,D)": {inner_label: 0.0 for inner_label in label_keys} for outer_label in label_keys}
                temp_avg_pyxD = {label: 0.0 for label in label_keys}

                ## Shuffle D
                df_D_shuffled = df_D.sample(frac=1, random_state=seed).reset_index(drop=True)
                D = "\n".join(
                    [f"{row['note']} <output>{row['label']}</output>\n" for _, row in df_D_shuffled.iterrows()]
                )
                # print("D:", D)
                
                # p(u|z,D)
                prompt_puzD = prompt.get_puzD_prompt(z, D)
                # print("\n########## <Prompt p(u|z,D)> ##########")
                # print(prompt_puzD)
                # print("########## <Prompt p(u|z,D)\> ##########")
                output_puzD, puzD = chat_tabular(prompt_puzD, label_keys, seed)
                # print("\n########## <Output p(u|z,D)\> ##########")
                # print(output_puzD)
                # print("########## <Output p(u|z,D)\> ##########")
                if not re.search(r'\d+</output>', output_puzD):
                    print("Output format not as expected for p(u|z,D), retrying with new seed...")
                    seed += 1
                    failed_seeds += 1
                    continue
                # print("\n########## <Probabilities p(u|z,D)\> ##########")
                # print(puzD)
                # print("########## <Probabilities p(u|z,D)\> ##########")
                if not puzD:
                    print("Empty probabilities detected (p(u|z,D)), retrying with new seed...")
                    seed += 1
                    failed_seeds += 1
                    continue

                # Accumulate into the temporary dictionary
                for label, prob in puzD.items():
                    temp_avg_puzD[label] += prob
                
                # p(y|x,u,z,D)
                skip_seed = False  # flag to skip the seed if any probability is empty
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
                    # print("\n########## <Prompt p(y|x,u,z,D)> ##########")
                    # print(prompt_pyxuzD)
                    # print("########## <Prompt p(y|x,u,z,D)\> ##########")
                    output_pyxuzD, pyxuzD = chat_tabular(prompt_pyxuzD, label_keys, seed)
                    # print("\n########## <Output p(y|x,u,z,D)\> ##########")
                    # print(output_pyxuzD)
                    # print("########## <Output p(y|x,u,z,D)\> ##########")
                    if not re.search(r'\d+</output>', output_pyxuzD):
                        print("Output format not as expected for p(y|x,u,z,D), retrying with new seed...")
                        skip_seed = True
                        failed_seeds += 1
                        break
                    # print(f"\n########## <p(y|x,u{u_value},z,D) Probabilities> ##########")
                    # print(pyxuzD)
                    # print(f"########## <p(y|x,u{u_value},z,D) Probabilities\> ##########")
                    if not pyxuzD:
                        print(f"Empty probabilities detected (p(y|x,u{u_value},z,D)), retrying with new seed...")
                        skip_seed = True
                        failed_seeds += 1
                        break
                    for label, prob in pyxuzD.items():
                        temp_avg_pyxuzD[f"p(y|x,u{u_value},z,D)"][label] += prob

                if skip_seed:
                    seed += 1
                    continue

                # p(y|x,D)
                prompt_pyxD = prompt.get_pyxD_prompt(x, D)
                # print("\n########## <Prompt p(y|x,D)> ##########")
                # print(prompt_pyxD)
                # print("########## <Prompt p(y|x,D)\> ##########")
                output_pyxD, pyxD = chat_tabular(prompt_pyxD, label_keys, seed)
                # print("\n########## <Output p(y|x,D)> ##########")
                # print(output_pyxD)
                # print("########## <Output p(y|x,D)\> ##########")
                if not re.search(r'\d+</output>', output_pyxD):
                    print("Output format not as expected for p(y|x,D), retrying with new seed...")
                    seed += 1
                    failed_seeds += 1
                    continue
                # print("\n########## <Probabilities p(y|x,D)> ##########")
                # print(pyxD)
                # print("########## <Probabilities p(y|x,D)\> ##########")
                if not pyxD:
                    print("Empty probabilities detected (p(y|x,D)), retrying with new seed...")
                    seed += 1
                    failed_seeds += 1
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
                # print(f"Successful Seeds: {successful_seeds}/{num_seeds}")
                successful_seeds_lst.append(seed)
                seed += 1

            # print(f"Successful Seeds: {successful_seeds_lst}")

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

            # print(f"\np(u|z,D) = {avg_puzD_probs}")
            # print(f"\np(y|x,u,z,D) = {avg_pyxuzD_probs}")
            # print(f"\np(y|x,z,D) = {avg_pyxzD_probs}")
            # print(f"\np(y|x,D) = {avg_pyxD_probs}")

            ## Thresholding p(y|x,z,D) and p(y|x,D) via KL Divergence
            kl_lst = []
            kl = calculate_kl_divergence(avg_pyxzD_probs, avg_pyxD_probs)
            kl_lst.append(kl)
            # print(f"\nKL divergence between p(y|x,z,D) and p(y|x,D): {kl}")

            kl_threshold = 0.2
            if kl < kl_threshold:
                # Compute Va
                H_pyxuzD = {}
                for key, probs in avg_pyxuzD_probs.items():
                    H_pyxuzD[f"H[{key}]"] = calculate_entropy(probs)
                E_H_pyxuzD = sum(H_pyxuzD[f"H[p(y|x,u{u_value},z,D)]"] * puzD[u_value] for u_value in puzD.keys())
                # print(f"\nVa = E_p(u|z,D)[H(p(y|x,u,z,D))] = {E_H_pyxuzD}")
                row['Va'] = E_H_pyxuzD
                min_Va_lst.append(row)
            
            if len(min_Va_lst) == 0:
                print(f"kl values:{kl_lst}")
                raise ValueError("check threshold!")

            # print("No. Rows before threshold", len(data_z))
            # print("No. Rows after threshold", len(min_Va_lst))
            
        # print(f"Failed Seeds: {failed_seeds}")
        # Entropy Calculations
        ## Compute Total Uncertainty
        H_pyxD = calculate_entropy(avg_pyxD_probs)

        ## Compute min aleatoric uncertainty, Va
        # if len(min_Va_lst) == 0:
            # print("No valid Va values found.")
            # min_Va = np.nan
        # else:
        min_Va = min(min_Va_lst, key=lambda row: row['Va'])
        min_Va['TU'] = H_pyxD

        # print(f"\nMin Va: {min_Va}")

        ## Compute epistemic uncertainty, Ve
        # print()
        # print(f"Ve = H[p(y|x,D)] - Va = H[p(y|x,D)] - E_p(u|z,D)[H[p(y|x,u,z,D)]]")
        # print(f"Total Uncertainty = H[p(y|x,D)] = {H_pyxD}")
        
        # if min_Va is np.nan:
            # print("No valid Va values found.")
            # Ve = np.nan
        # else:
            # print(f"min Va = {min_Va['Va']}")
        Ve = H_pyxD - min_Va['Va']
        min_Va['Ve'] = Ve
            
        # print(f"Ve = {Ve}")

        # print(x_row)
        # print(min_Va)
        # Example x_row and min_Va (assuming they are pandas Series)

        x_z = {
            'x_note': x_row['note'],
            f'x_{perturb_x}': x_row[perturb_x],
            'TU': min_Va['TU'],
            'Va': min_Va['Va'],
            'Ve': min_Va['Ve']
        }
        x_z = pd.DataFrame([x_z])
        x_z_lst.append(x_z)

    df_plot = pd.concat(x_z_lst, ignore_index=True)
    df_plot.to_csv(f"df_plot_{perturb_x}.csv", index=False)

    plt.figure(figsize=(12, 10))
    plt.scatter(df_plot[f"x_{perturb_x}"], df_plot["TU"], label="Total Uncertainty (TU)", s=100)
    plt.scatter(df_plot[f"x_{perturb_x}"], df_plot["Va"], label="Minimum Va (Aleatoric Uncertainty)", s=100)
    for _, row in df_D.iterrows():
        work_hours = row["Work hours per week"]
        label = row["label"]
        
        # Color coding: Blue for label 0, Red for label 1
        color = "blue" if label == 0 else "red"
        
        plt.axvline(x=work_hours, color=color, linestyle="--", alpha=0.5, linewidth=1)
    plt.xlabel(f"{perturb_x}")
    plt.ylabel("Uncertainty")
    plt.title(f"Total Uncertainty and Minimum Aleatoric Uncertainty (Va) by {perturb_x}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"tabular_plt_{perturb_x}.pdf")

if __name__ == "__main__":
    # Argument Parser
    pd.set_option('display.max_columns', None)
    parser = argparse.ArgumentParser(description='Run VPUD')
    parser.add_argument("--seed", default=123)
    parser.add_argument("--data_path", default="datasets_tabular/adult")
    parser.add_argument("--num_seeds", default=5)
    parser.add_argument("--perturb_x", default="Work hours per week")
    args = parser.parse_args()

    main()
