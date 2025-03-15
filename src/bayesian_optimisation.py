import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize, Normalize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf

def new_candidate(z_values: list[float], maximisation_quantity: list[float], lower_bound: float, upper_bound: float) -> float:
    """
    Given the current z values and their corresponding entropy values,
    this function returns the new candidate z value to evaluate next.
    
    Args:
    z_values: list[float] - the current z values
    entropy_values: list[float] - the corresponding entropy values
    
    Returns:
    float - the new candidate z value to evaluate
    """
    # Create the training data
    train_X = torch.tensor(z_values, dtype=torch.float64).unsqueeze(-1)
    train_Y = torch.tensor(maximisation_quantity, dtype=torch.float64).unsqueeze(-1)
    
    # Create the model
    gp = SingleTaskGP(
        train_X=train_X, 
        train_Y=train_Y,
        input_transform=Normalize(d=1),
        outcome_transform=Standardize(m=1)
        )
    
    # Fit the model
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    
    # Create the acquisition function
    EI = LogExpectedImprovement(gp, best_f=train_Y.max())
    
    # Optimize the acquisition function
    candidate, acq_value = optimize_acqf(
        acq_function=EI,
        bounds=torch.tensor([[lower_bound], [upper_bound]]),
        q=1,
        num_restarts=5,
        raw_samples=20,
    )
    
    return candidate.item()