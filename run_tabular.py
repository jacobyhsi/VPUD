import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.dataset import load_dataset
from src.prompt import Prompt
from src.chat import chat_tabular
from src.utils import calculate_entropy, calculate_kl_divergence, TabularUtils, ToyDataUtils

# note to self: play around with num_seeds and z_samples

# Main
def main():
    global_seed = int(args.seed)
    perturb_x = args.perturb_x
    new_feature_col_names: list[str] = args.new_feature_col_names
    perturb_x = "iris"

    # Load dataset
    # data, test, label_keys = load_dataset(args.data_path)

    data, test, label_keys = load_dataset("datasets_tabular/iris")

    # Sampling D as icl
    num_D = args.num_D
    df_D = data.sample(n=num_D, random_state=global_seed)
    data = data.drop(df_D.index)
    feature_columns = ToyDataUtils.get_feature_columns(data)
    if len(new_feature_col_names) > 0:
        feature_column_map = {col: new_col for col, new_col in zip(feature_columns, new_feature_col_names)}
        df_D.rename(columns=feature_column_map, inplace=True)
        df_D["note"] = df_D.apply(lambda row: TabularUtils.build_note(row), axis=1)
        test.rename(columns=feature_column_map, inplace=True)
        test["note"] = test.apply(lambda row: TabularUtils.build_note(row), axis=1)
    df_D.to_csv(f"results_tabular/iris/df_D_{perturb_x}_{args.run_name}.csv", index=False)
    # Sampling x
    # x_row = test.sample(n=1, random_state=global_seed)
    # test = test.drop(x_row.index) # drop the sampled x
    # x = x_row['note'].iloc[0]

    data_x = test.sample(n=args.num_x_values, random_state=global_seed)
    test = test.drop(data_x.index) # drop the sampled x

    # perturb x for visualization 
    # if perturb_x != 'all':
    #     data_x = TabularUtils.perturb_x(data, x_row, perturb_x)
    # else:
    #     data_x = TabularUtils.perturb_all_x(data, x_row, df_D)
    # Perturb x

    # Sample z
    df_z = data.sample(n=1, random_state=global_seed)
    data = data.drop(df_z.index)
    z = df_z['note'].iloc[0]
    # print("z:", z)

    prompt = Prompt(prompt_type="tabular")
    
    x_z_lst = []
    failed_seeds = 0

    # Processing x perturbations
    for i, x_row in tqdm(data_x.iterrows(), total=len(data_x), desc="Processing x perturbations"):
        # print(f"\nProcessing x pertubation: {i}/{len(data_x)}")
        x = x_row['note']
        print(f"\nx: {x}")
        # Processing z Probabilities
        min_Va_lst = []
        seed = 0

        # perturbed z should be close to x.
        # data_z = TabularUtils.perturb_all_z(data, df_z, df_D)
        data_z = TabularUtils.perturb_z(data=df_D, x_row=x_row, z_samples=10)

        for i, row in tqdm(data_z.iterrows(), total=len(data_z), desc="Processing z perturbations"):
            z = row['note']
            print(f"\nz: {z}")

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
                seed += 1
                # print("\n########## <Output p(u|z,D)\> ##########")
                # print(output_puzD)
                # print("########## <Output p(u|z,D)\> ##########")

                if not re.search(r'\d+</output>', output_puzD):
                    print(f"Output format not as expected for p(u|z,D): {output_puzD}, retrying with new seed...")
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
                    seed += 1
                    # print("\n########## <Output p(y|x,u,z,D)\> ##########")
                    # print(output_pyxuzD)
                    # print("########## <Output p(y|x,u,z,D)\> ##########")
                    if not re.search(r'\d+</output>', output_pyxuzD):
                        print(f"Output format not as expected for p(y|x,u,z,D): {output_pyxuzD}, retrying with new seed...")
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
                seed += 1
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

        # kl_threshold = 0.2
        # if kl < kl_threshold:
        #     # Compute Va
        #     H_pyxuzD = {}
        #     for key, probs in avg_pyxuzD_probs.items():
        #         H_pyxuzD[f"H[{key}]"] = calculate_entropy(probs)
        #     E_H_pyxuzD = sum(H_pyxuzD[f"H[p(y|x,u{u_value},z,D)]"] * puzD[u_value] for u_value in puzD.keys())
        #     # print(f"\nVa = E_p(u|z,D)[H(p(y|x,u,z,D))] = {E_H_pyxuzD}")
        #     row['Va'] = E_H_pyxuzD
        #     min_Va_lst.append(row)
        
        # if len(min_Va_lst) == 0:
        #     print(f"kl values:{kl_lst}")
        #     raise ValueError("check threshold!")

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

        pred_label = max(avg_pyxD_probs, key=avg_pyxD_probs.get)
        true_label = x_row['label']
            
        # print(f"Ve = {Ve}")

        # print(x_row)
        # print(min_Va)
        # Example x_row and min_Va (assuming they are pandas Series)
        if perturb_x != 'all':
            x_z = {
                'x_note': x_row['note'],
                # f'x_{perturb_x}': x_row[perturb_x],
                'TU': min_Va['TU'],
                'Va': min_Va['Va'],
                'Ve': min_Va['Ve'],
                'true_label': true_label,
                'pred_label': pred_label,
            }
        else:
            x_z = {
                'x_note': x_row['note'],
                # 'radius': x_row['radius'],
                'TU': min_Va['TU'],
                'Va': min_Va['Va'],
                'Ve': min_Va['Ve'],
                'true_label': true_label,
                'pred_label': pred_label,
            }
        x_z = pd.DataFrame([x_z])
        x_z_lst.append(x_z)

        df_plot = pd.concat(x_z_lst, ignore_index=True)
        df_plot.to_csv(f"results_tabular/iris/df_plot_{perturb_x}_{args.run_name}.csv", index=False)

if __name__ == "__main__":
    # Argument Parser
    pd.set_option('display.max_columns', None)
    parser = argparse.ArgumentParser(description='Run VPUD')
    parser.add_argument("--seed", default=123)
    parser.add_argument("--data_path", default="datasets_tabular/adult")
    parser.add_argument("--num_seeds", default=5)
    parser.add_argument("--perturb_x", default="all")
    parser.add_argument("--run_name", default="test")
    parser.add_argument("--num_D", default=25, type=int)
    parser.add_argument("--num_x_values", default=10, type=int)
    parser.add_argument("--new_feature_col_names", nargs="+", default=[])
    args = parser.parse_args()
    main()
