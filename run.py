import argparse
import pandas as pd
from src.dataset import *
from src.prompt import *
from src.chat import *
from src.utils import *


# Main
def main():
    # Load dataset
    data, test, label_name, label_map, label_keys = load_dataset(args.data_path)

    # Sampling x
    x_row = data.sample(n=1, random_state=args.seed)
    data = data.drop(x_row.index) # drop the sampled x
    x = x_row['note'].iloc[0]
    print("x:", x)
    x_y = x_row['label'].iloc[0]
    print("x label:", x_y)

    # Sampling D as icl
    num_D = 3
    df_D = data.sample(n=num_D, random_state=args.seed)
    data = data.drop(df_D.index)

    # Sample z
    df_z = data.sample(n=1, random_state=args.seed)
    data = data.drop(df_z.index)
    z = df_z['note'].iloc[0]
    print("z:", z)
    data_z = pertube_z(data, df_z, z_samples=10) # z_samples number of pertubations per z

    seed_num = 3
    prompt = Prompt(label_name, label_map, label_keys)
    
    min_Va_lst = []
    # Processing z Probabilities
    for i, row in data_z.iterrows():
        z = row['note']
        z_y = row['label']

        # Initialize dictionaries to store average probabilities
        avg_puzD_probs = {label: 0.0 for label in label_keys}
        avg_pyxuzD_probs = {f"p(y|x,u{outer_label},z,D)": {inner_label: 0.0 for inner_label in label_keys} for outer_label in label_keys}
        avg_pyxzD_probs = {label: 0.0 for label in label_keys}
        avg_pyxD_probs = {label: 0.0 for label in label_keys}

        # Extracting prompt probabilities: p(u|z,D), p(y|x,u,z,D), p(y|x,D)
        successful_seeds = 0
        successful_seeds_lst = []
        seed = 0
        seed_num = 5  # target number of successful seeds
        while successful_seeds < seed_num:
            print(f"\nSeed {seed + 1}")
            print(f"Successful Seeds: {successful_seeds}/{seed_num}")

            ## Shuffle D
            df_D_shuffled = df_D.sample(frac=1, random_state=seed).reset_index(drop=True)
            D = "\n".join(
                [f"{row['note']} <output>{row['label']}</output>\n" for _, row in df_D_shuffled.iterrows()]
            )
            print("D:", D)
            
            # p(u|z,D)
            prompt_puzD = prompt.get_puzD_prompt(z, D)
            print("\n########## <Prompt p(u|z,D)> ##########")
            print(prompt_puzD)
            print("########## <Prompt p(u|z,D)\> ##########")
            output_puzD, puzD = chat(prompt_puzD, label_keys, seed)
            print("\n########## <Output p(u|z,D)\> ##########")
            print(output_puzD)
            print("########## <Output p(u|z,D)\> ##########")
            if not re.search(r'\d+</output>', output_puzD):
                print("Output format not as expected for p(u|z,D), retrying with new seed...")
                seed += 1
                continue
            print("\n########## <Probabilities p(u|z,D)\> ##########")
            print(puzD)
            print("########## <Probabilities p(u|z,D)\> ##########")
            if not puzD:
                print("Empty probabilities detected (p(u|z,D)), retrying with new seed...")
                seed += 1
                continue

            for label, prob in puzD.items():
                avg_puzD_probs[label] += prob
            
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
                print("\n########## <Prompt p(y|x,u,z,D)> ##########")
                print(prompt_pyxuzD)
                print("########## <Prompt p(y|x,u,z,D)\> ##########")
                output_pyxuzD, pyxuzD = chat(prompt_pyxuzD, label_keys, seed)
                print("\n########## <Output p(y|x,u,z,D)\> ##########")
                print(output_pyxuzD)
                print("########## <Output p(y|x,u,z,D)\> ##########")
                if not re.search(r'\d+</output>', output_pyxuzD):
                    print("Output format not as expected for p(y|x,u,z,D), retrying with new seed...")
                    skip_seed = True
                    break
                print(f"\n########## <p(y|x,u{u_value},z,D) Probabilities> ##########")
                print(pyxuzD)
                print(f"########## <p(y|x,u{u_value},z,D) Probabilities\> ##########")
                if not pyxuzD:
                    print(f"Empty probabilities detected (p(y|x,u{u_value},z,D)), retrying with new seed...")
                    skip_seed = True
                    break
                for label, prob in pyxuzD.items():
                    avg_pyxuzD_probs[f"p(y|x,u{u_value},z,D)"][label] += prob

            if skip_seed:
                seed += 1
                continue

            # p(y|x,D)
            prompt_pyxD = prompt.get_pyxD_prompt(x, D)
            print("\n########## <Prompt p(y|x,D)> ##########")
            print(prompt_pyxD)
            print("########## <Prompt p(y|x,D)\> ##########")
            output_pyxD, pyxD = chat(prompt_pyxD, label_keys, seed)
            print("\n########## <Output p(y|x,D)> ##########")
            print(output_pyxD)
            print("########## <Output p(y|x,D)\> ##########")
            if not re.search(r'\d+</output>', output_pyxD):
                print("Output format not as expected for p(y|x,D), retrying with new seed...")
                seed += 1
                continue
            print("\n########## <Probabilities p(y|x,D)> ##########")
            print(pyxD)
            print("########## <Probabilities p(y|x,D)\> ##########")
            if not pyxD:
                print("Empty probabilities detected (p(y|x,D)), retrying with new seed...")
                seed += 1
                continue

            for label, prob in pyxD.items():
                avg_pyxD_probs[label] += prob

            successful_seeds += 1
            print(f"Successful Seeds: {successful_seeds}/{seed_num}")
            successful_seeds_lst.append(seed)
            seed += 1
        
        print(f"Successful Seeds: {successful_seeds_lst}")

        # Average probabilities
        for label in label_keys:
            avg_puzD_probs[label] /= seed_num
            avg_pyxD_probs[label] /= seed_num
            for u_label in label_keys:
                avg_pyxuzD_probs[f"p(y|x,u{u_label},z,D)"][label] /= seed_num

        # p(y|x,z,D) via marginalization
        for label in label_keys:
            avg_pyxzD_probs[label] = sum(
                avg_pyxuzD_probs[f"p(y|x,u{u_label},z,D)"][label] * avg_puzD_probs[u_label]
                for u_label in avg_puzD_probs.keys()
            )

        print(f"\np(u|z,D) = {avg_puzD_probs}")
        print(f"\np(y|x,u,z,D) = {avg_pyxuzD_probs}")
        print(f"\np(y|x,z,D) = {avg_pyxzD_probs}")
        print(f"\np(y|x,D) = {avg_pyxD_probs}")

        ## Thresholding p(y|x,z,D) and p(y|x,D) via KL Divergence
        kl = kl_divergence(avg_pyxzD_probs, avg_pyxD_probs)
        print(f"\nKL divergence between p(y|x,z,D) and p(y|x,D): {kl}")

        if kl < 0.01:
            # Compute Va
            H_pyxuzD = {}
            for key, probs in avg_pyxuzD_probs.items():
                H_pyxuzD[f"H[{key}]"] = calculate_entropy(probs)
            E_H_pyxuzD = sum(H_pyxuzD[f"H[p(y|x,u{u_value},z,D)]"] * puzD[u_value] for u_value in puzD.keys())
            print(f"\nVa = E_p(u|z,D)[H(p(y|x,u,z,D))] = {E_H_pyxuzD}")
            row['Va'] = E_H_pyxuzD
            min_Va_lst.append(row)

        print("No. Rows before threshold", len(data_z))
        print("No. Rows after threshold", len(min_Va_lst))

    # Entropy Calculations
    ## Compute Total Uncertainty
    H_pyxD = calculate_entropy(avg_pyxD_probs)
    print(f"\nTotal Uncertainty = H(p(y|x,D)) = {H_pyxD}")

    ## Compute min aleatoric uncertainty, Va
    min_Va = min(min_Va_lst, key=lambda row: row['Va'])
    print(f"\nMin Va: {min_Va}")

    ## Compute epistemic uncertainty, Ve
    print()
    print(f"Ve = H[p(y|x,D)] - Va = H[p(y|x,D)] - E_p(u|z,D)[H[p(y|x,u,z,D)]]")
    print(f"Total Uncertainty = H[p(y|x,D)] = {H_pyxD}")
    print(f"min Va = {min_Va['Va']}")
    Ve = H_pyxD - min_Va['Va']
    print(f"Ve = {Ve}")


if __name__ == "__main__":
    # Argument Parser
    pd.set_option('display.max_columns', None)
    parser = argparse.ArgumentParser(description='Run VPUD')
    parser.add_argument("--seed", default=123)
    parser.add_argument("--data_path", default="datasets_serialized/adult")
    args = parser.parse_args()

    main()
