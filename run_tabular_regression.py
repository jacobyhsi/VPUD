import re
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.dataset import TabularDataset
from src.prompt import Prompt
from src.chat import chat_response_only
from src.utils import calculate_entropy, calculate_kl_divergence, TabularUtils, GaussianDistribution, ToyRegressionUtils, extract

from typing import Optional


# note to self: play around with num_seeds and z_samples
def calculate_gaussian(
    prompt: str,
    current_seed: int,
    num_permutations: Optional[int]=10,
    num_outlier_pairs_to_remove: Optional[int]=1,
    model_name: str="Qwen/Qwen2.5-14B",
):
    # Samples from the distribution
    distribution_samples = []

    successful_seeds = 0
    attempts = 0
    while successful_seeds < num_permutations + num_outlier_pairs_to_remove*2 and attempts < 100:

        try:
            permutation_seed = current_seed

            # print(f"Prompt:")
            # print(prompt)
            
            # Get the prediction and probabilities from the model
            response = chat_response_only(prompt, seed=permutation_seed, model=model_name)
            
            attempts += 1  
            current_seed += 1      

            sample = extract(response)
            
            if not isinstance(sample, float|int):
                print(f"Invalid sample: {sample}")
                raise ValueError(f"Invalid sample: {sample}")
            
            # print(f"y_sample: {sample}")
            
            distribution_samples.append(sample)

            successful_seeds += 1
            
        except:
            print(f"Attempt {attempts} failed. Restarting for seed {successful_seeds}")   
            current_seed += 1
                    
    if successful_seeds == 0:
        raise ValueError(f"All seeds failed.")      
    
    gaussian = ToyRegressionUtils.gaussian_from_samples(distribution_samples, num_outlier_pairs_to_remove)
    
    # print(f"\nGaussian Approximation: mean = {gaussian.mean}, std = {gaussian.std}")
        
    return gaussian, distribution_samples, current_seed

# Main
def main():
    global_seed = int(args.seed)

    args.id_dataset = "lenses"

    if args.id_dataset == "iris":
        dataset_id = 53
        dataset_label = "class"
    if args.id_dataset == "lenses":
        dataset_id = 58
        dataset_label = "class"
    elif args.id_dataset == "estate":
        dataset_id = 477
        dataset_label = "Y house price of unit area"

    if args.ood_dataset == "beijing":
        dataset_ood = 381
    elif args.ood_dataset == "diabetes":
        dataset_ood = 296

    tabular_dataset = TabularDataset(dataset_id, global_seed)
    train_id, test_id, _ = tabular_dataset.load_data(config_path=f"datasets_tabular/{args.id_dataset}" ,label=dataset_label)
    train_ood, test_ood = tabular_dataset.generate_ood(dataset_ood)

    print("len(train_id), len(test_id):", len(train_id), len(test_id))

    # Switch between OOD and ID
    if args.distb == "OOD":
        train = train_ood
        test = test_ood
    else:
        train = train_id
        test = test_id

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

    num_seeds = args.num_seeds  # target number of successful seeds
    num_outlier_pairs_to_remove = args.num_outlier_pairs_to_remove
    
    # Processing x perturbations
    for i, x_row in tqdm(data_x.iterrows(), total=len(data_x), desc="Processing x perturbations"):
        # print(f"\nProcessing x pertubation: {i}/{len(data_x)}")
        x = x_row['note']
        # print(f"\nx: {x}")
        # Processing z Probabilities
        min_Va_lst = []
        seed = 0

        ## Shuffle D
        df_D_shuffled = df_D.sample(frac=1, random_state=seed).reset_index(drop=True)
        D = "\n".join(
            [f"{row['note']} <output> {row['label']} </output>\n" for _, row in df_D_shuffled.iterrows()]
        )
        # print("D:", D)
        
        # p(y|x,D)
    
        prompt_pyxD = prompt.get_pyxD_prompt(x, D)
        # print("\n########## <Prompt p(y|x,D)> ##########")
        # print(prompt_pyxD)
        # print("########## <Prompt p(y|x,D)\> ##########")
        pyxD, y_samples, seed = calculate_gaussian(
            prompt=prompt_pyxD,
            current_seed=seed,
            num_permutations=num_seeds,
            num_outlier_pairs_to_remove=num_outlier_pairs_to_remove
        )
        
        # print("\n########## <Output p(y|x,D)> ##########")
        # print(y_samples)
        # print("########## <Output p(y|x,D)\> ##########")

        # print("\n########## <Probabilities p(y|x,D)> ##########")
        # print(f"mean: {pyx.mean}, std: {pyx.std}")
        # print("########## <Probabilities p(y|x,D)\> ##########")

        data_z = TabularUtils.perturb_z(data=df_D, x_row=x_row, z_samples=num_z)
        for i, row in tqdm(data_z.iterrows(), total=len(data_z), desc="Processing z perturbations"):
            z = row['note']
            # print(f"\nz: {z}")

            # Extracting prompt probabilities: p(u|z,D), p(y|x,u,z,D), p(y|x,D)
            
            num_seeds = args.num_seeds  # target number of successful seeds
            # print(f"\nSample: {i}; Seed: {seed + 1}")
            # print(f"Successful Seeds: {successful_seeds}/{num_seeds}")

            ## Shuffle D
            df_D_shuffled = df_D.sample(frac=1, random_state=seed).reset_index(drop=True)
            D = "\n".join(
                [f"{row['note']} <output> {row['label']} </output>\n" for _, row in df_D_shuffled.iterrows()]
            )
            # print("D:", D)
            
            # p(u|z,D)
            prompt_puzD = prompt.get_puzD_prompt(z, D)
            # print("\n########## <Prompt p(u|z,D)> ##########")
            # print(prompt_puzD)
            # print("########## <Prompt p(u|z,D)\> ##########")
            puzD, u_samples, seed = calculate_gaussian(
                prompt=prompt_puzD,
                current_seed=seed,
                num_permutations=num_seeds,
                num_outlier_pairs_to_remove=num_outlier_pairs_to_remove
            )
            # print("\n########## <Output p(u|z,D)\> ##########")
            # print(f"u_samples: {u_samples}")
            # print("########## <Output p(u|z,D)\> ##########")

            # print("\n########## <Probabilities p(u|z,D)\> ##########")
            # print(f"mean: {puzD.mean}, std: {puzD.std}")
            # print("########## <Probabilities p(u|z,D)\> ##########")
                
            # p(y|x,u,z,D)
            dict_uz = {}
            for i, u_sample in enumerate(u_samples):
                df_copy = df_z.copy()
                df_copy["label"] = u_sample
                dict_uz[f"u{i}z"] = df_copy

            dict_uzD = {}
            for key, df_uz in dict_uz.items():
                dict_uzD[f"{key}D"] = pd.concat([df_uz, df_D], ignore_index=True)
                # Shuffle u,z,D
                dict_uzD[f"{key}D"] = dict_uzD[f"{key}D"].sample(frac=1, random_state=seed).reset_index(drop=True)

            prompt_uzD = {}
            for key, df_uzD in dict_uzD.items():
                prompt_uzD[f"{key}"] = "\n".join([f"{row['note']} <output> {row['label']} </output>\n" for _, row in df_uzD.iterrows()])

            HyxuzD = []
            pyxuzD_distributions: list[GaussianDistribution] = []
            for key, icl in prompt_uzD.items():
                prompt_pyxuzD = prompt.get_pyxuzD_prompt(x, icl)
                # print("\n########## <Prompt p(y|x,u,z,D)> ##########")
                # print(prompt_pyxuzD)
                # print("########## <Prompt p(y|x,u,z,D)\> ##########")
                pyxuzD, yxuzD_samples, seed = calculate_gaussian(prompt=prompt_pyxuzD, current_seed=seed, num_permutations=num_seeds, num_outlier_pairs_to_remove=num_outlier_pairs_to_remove)
                HyxuzD.append(pyxuzD.entropy)
                pyxuzD_distributions.append(pyxuzD)
                
                # print("\n########## <Output p(y|x,u,z,D)\> ##########")
                # print(yxuzD_samples)

                # print(f"\n########## <p(y|x,u{u_value},z,D) Probabilities> ##########")
                # print(f"mean: {pyxuzD.mean}, std: {pyxuzD.std}")
                # print(f"########## <p(y|x,u{u_value},z,D) Probabilities\> ##########")

            # p(y|x,z,D) via marginalization
            pyxzD_samples = []
            for _ in range(100):
                u_sample = np.random.randint(len(u_samples))   
                pyxzD_sample = pyxuzD_distributions[u_sample].sample()    
                pyxzD_samples.append(pyxzD_sample)        
            pyxzD = ToyRegressionUtils.gaussian_from_samples(pyxzD_samples)

            HyxzD = np.mean(HyxuzD)
            
            ## Thresholding p(y|x,z,D) and p(y|x,D) via KL Divergence
            kl = ToyRegressionUtils.calculate_kl_divergence(pyxD, pyxzD)
            data_z.at[i, 'KL'] = kl
            data_z.at[i, 'HyxzD'] = np.round(HyxzD, 5)

        min_kl = data_z.sort_values('KL').head(5)
        for _, row in min_kl.iterrows():

            E_H_pyxuzD = row['HyxzD']

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
        H_pyxD = np.round(pyxD.entropy,5)

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

        pred_label = pyxD.mean
        true_label = x_row['label']
            
        x_z = {
            'x_note': x_row['note'],
            # f'x_{args.distb}': x_row[args.distb],
            'TU': min_Va['TU'],
            'Va': min_Va['Va'],
            'Ve': min_Va['Ve'],
            'true_label': true_label,
            'pred_label': pred_label,
            'pred_std': pyxD.std,
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
    parser.add_argument("--num_seeds", default=5, type=int)
    parser.add_argument("--distb", default="OOD")
    parser.add_argument("--num_outlier_pairs_to_remove", default=1, type=int)
    args = parser.parse_args()
    
    main()