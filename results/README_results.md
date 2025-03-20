## Description of Results

### Logistic Regression 3
- Experiment 1:
    - Summary: Bayesian optimisation (with 5 initial random z, 10 z from BO); 15 shots
    - Run script: `python run_regression.py --x_range "{'x1': [-15, 15, 0.2]}" --shots 15 --num_modified_z 15 --num_random_z 5 --save_directory experiment_1 --run_name 15_shot_15_z_5_random_10_perm --num_permutations 10`
- Experiment 2:
    - Summary: Small perturbations of x to obtain z (15 z); 15 shot
    - Run script: `python run_regression.py --x_range "{'x1': [-15, 15, 0.2]}" --shots 15 --num_modified_z 15 --num_random_z 15 --save_directory experiment_2 --run_name 15_shot_15_z_5_random_10_perm --num_permutations 10 --perturbation_std 0.1` 

### Moons
- Experiment_1:
    - Summary: Bayesian optimisation (with 5 initial random z, 10 z from BO); 15 shots
    - `python run_regression.py --x_range "{'x1': [-1, 2, 0.2], 'x2': [-1, 2, 0.2]}" --shots 15 --num_modified_z 15 --num_random_z 5 --save_directory experiment_moons_1 --run_name 15_shot_15_z_5_random_10_perm --num_permutations 10 --decimal_places 2 --dataset_name moons`
- Experiment 2:
    - Small perturbations
    - Summary: Small perturbations of x to obtain z (15 z); 15 shot
    - `python run_regression.py --x_range "{'x1': [-1, 2, 0.2], 'x2': [-1, 2, 0.2]}" --shots 15 --num_modified_z 15 --num_random_z 15 --save_directory experiment_moons_2 --run_name 15_shot_15_z_15_random_10_perm --num_permutations 10 --decimal_places 2 --dataset_name moons --perturbation_std 0.1`
- Experiment 3:
    - Summary: Bayesian optimisation (with 5 initial random z, 10 z from BO); 30 shots
    - Run script: `python run_regression.py --x_range "{'x1': [-1.5, 2.5, 0.2], 'x2': [-1, 2, 0.2]}" --shots 30 --num_modified_z 15 --num_random_z 5 --save_directory experiment_moons_3 --run_name 30_shot_15_z_5_random_10_perm --num_permutations 10 --decimal_places 2 --dataset_name moons`
- Experiment 4:
    - Summary: Small perturbations of x to obtain z (15 z); 30 shot
    - Run scripts: `python run_regression.py --x_range "{'x1': [-1.5, 2.5, 0.2], 'x2': [-1, 2, 0.2]}" --shots 30 --num_modified_z 15 --num_random_z 15 --save_directory experiment_moons_4 --run_name 30_shot_15_z_15_random_10_perm --num_permutations 10 --decimal_places 2 --dataset_name moons --perturbation_std 0.1`