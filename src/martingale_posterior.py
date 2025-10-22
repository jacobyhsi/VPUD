import re
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def parse_response(response, feature_columns:list[str], label_tag_name="output", dataset_type:str="toy_classification") -> None | pd.DataFrame:
    """
    Parse the response from a model into a DataFrame. If the response is in wrong format, further data is not returned.
    
    e.g. " x1 = 0.5 <output>1</output>\n x1 = 0.7 <output>0</output>\n" -> pd.DataFrame({'x1': [0.5, 0.7], 'label': [1, 0]})
    
    Use regex
    """
    if not isinstance(response, str):
        return None
    lines = response.strip().split('\n')
    data = {col: [] for col in feature_columns + ["label"]}

    for line in lines:
        match = re.match(rf'^(.*?)<{label_tag_name}>(.*?)</{label_tag_name}>$', line.strip())
        if not match:
            break
        feature_parts, label_part = match.groups()
        row = {}
        try:
            for feature_part in feature_parts.split(';'):
                for col in feature_columns:
                    col_match = re.search(rf'{col} = (.*?)(?=\s|$)', feature_part)
                    if col_match:
                        row.update({col: float(col_match.group(1))})
        except:
            break
        try:
            if dataset_type == 'toy_classification':
                if not float(label_part.strip()).is_integer():
                    break
                else:
                    row['label'] = float(label_part.strip())
            elif dataset_type == 'toy_regression':
                row['label'] = float(label_part.strip())
        except:
            break
        if len(row) == len(feature_columns) + 1 and all(col in row for col in feature_columns):  # Ensure all features are present
            for col in feature_columns:
                data[col].append(row[col])
            data['label'].append(row['label'])
        else:
            break

    return pd.DataFrame(data)

def logistic_regression_negative_log_likelihood(X: torch.Tensor, y: torch.Tensor, b0: torch.Tensor, b1: torch.Tensor):
    """
    Compute the negative log likelihood of the logistic regression model.
    
    Args:
        X (torch.Tensor): Input features.
        y (torch.Tensor): Target labels.
        b0 (torch.Tensor): Intercept term.
        b1 (torch.Tensor): Coefficients for the features.
        
    Returns:
        torch.Tensor: Log likelihood value.
    """
    logits = b0 + X @ b1
    log_likelihood = torch.sum(y * (logits + 1e-5) - torch.log(1 + torch.exp(logits + 1e-5)))
    return -log_likelihood

def probit_regression_negative_log_likelihood(X: torch.Tensor, y: torch.Tensor, b0: torch.Tensor, b1: torch.Tensor):
    """
    Compute the negative log likelihood of the probit regression model.
    
    Args:
        X (torch.Tensor): Input features.
        y (torch.Tensor): Target labels.
        b0 (torch.Tensor): Intercept term.
        b1 (torch.Tensor): Coefficients for the features.
        
    Returns:
        torch.Tensor: Log likelihood value.
    """
    logits = b0 + X @ b1
    probit = torch.distributions.Normal(0, 1).cdf(logits)
    log_likelihood = torch.sum(y * torch.log(probit + 1e-5) + (1 - y) * torch.log(1 - probit + 1e-5))
    return -log_likelihood

def gaussian_negative_log_likelihood(X: torch.Tensor, y: torch.Tensor, b0: torch.Tensor, b1: torch.Tensor, sigma: torch.Tensor):
    """
    Compute the negative log likelihood of the Gaussian regression model.
    
    Args:
        X (torch.Tensor): Input features.
        y (torch.Tensor): Target labels.
        b0 (torch.Tensor): Intercept term.
        b1 (torch.Tensor): Coefficients for the features.
        sigma (torch.Tensor): Standard deviation of the Gaussian noise.

    Returns:
        torch.Tensor: Log likelihood value.
    """
    preds = b0 + X @ b1
    dist = torch.distributions.Normal(preds, sigma)
    log_likelihood = torch.sum(dist.log_prob(y))
    return -log_likelihood

def find_maximum_likelihood_estimates(X: torch.Tensor, y: torch.Tensor):
    """
    Find the maximum likelihood estimates for the logistic regression parameters.
    
    Args:
        X (torch.Tensor): Input features.
        y (torch.Tensor): Target labels.
        
    Returns:
        tuple: Estimated parameters (b0, b1).
    """
    b0 = torch.tensor(0.0, requires_grad=True, dtype=torch.float32)
    b1 = torch.tensor(0.01*np.random.randn(X.shape[1]) / np.sqrt((X.shape[1] + 1)), requires_grad=True, dtype=torch.float32)

    optimizer = torch.optim.AdamW([b0, b1], lr=0.0001)
    
    num_epochs = 10000
    for _ in range(num_epochs):
        optimizer.zero_grad()
        loss = logistic_regression_negative_log_likelihood(X, y, b0, b1)
        loss.backward()
        optimizer.step()

    return b0.item(), b1.tolist()

def find_maximum_likelihood_estimates_probit(X: torch.Tensor, y: torch.Tensor):
    """
    Find the maximum likelihood estimates for the probit regression parameters.
    
    Args:
        X (torch.Tensor): Input features.
        y (torch.Tensor): Target labels.
        
    Returns:
        tuple: Estimated parameters (b0, b1).
    """
    b0 = torch.tensor(0.0, requires_grad=True, dtype=torch.float32)
    b1 = torch.tensor(np.random.randn(X.shape[1]), requires_grad=True, dtype=torch.float32)
    
    optimizer = torch.optim.AdamW([b0, b1], lr=0.1)
    
    num_epochs = 10000
    for _ in range(num_epochs):
        optimizer.zero_grad()
        loss = probit_regression_negative_log_likelihood(X, y, b0, b1)
        loss.backward()
        optimizer.step()

    return b0.item(), b1.tolist()

def find_maximum_likelihood_estimates_gaussian(X: torch.Tensor, y: torch.Tensor):
    """
    Find the maximum likelihood estimates for the Gaussian regression parameters.

    Args:
        X (torch.Tensor): Input features.
        y (torch.Tensor): Target labels.

    Returns:
        tuple: Estimated parameters (b0, b1, sigma).
    """
    b0 = torch.tensor(0.0, requires_grad=True, dtype=torch.float32)
    b1 = torch.tensor(0.01*np.random.randn(X.shape[1]) / np.sqrt((X.shape[1] + 1)), requires_grad=True, dtype=torch.float32)
    sigma = torch.tensor(1.0, requires_grad=True, dtype=torch.float32)

    optimizer = torch.optim.AdamW([b0, b1, sigma], lr=0.1)

    num_epochs = 10000
    
    for _ in range(num_epochs):
        optimizer.zero_grad()
        loss = gaussian_negative_log_likelihood(X, y, b0, b1, sigma)
        loss.backward()
        optimizer.step()

    return b0.item(), b1.tolist(), sigma.item()

###

def quadratic_featurizer(X: torch.Tensor) -> torch.Tensor:
    """
    Generate quadratic features for the input tensor X.
    
    Args:
        X (torch.Tensor): Input features.
        
    Returns:
        torch.Tensor: Tensor with quadratic features added.
    """
    i, j = torch.triu_indices(X.shape[1], X.shape[1])
    cross_features = X[:, i] * X[:, j]
    return torch.cat([X, cross_features], dim=1)

def cubic_featurizer(X: torch.Tensor) -> torch.Tensor:
    """
    Generate cubic (and quadratic) features for the input tensor X.

    Args:
        X (torch.Tensor): Input features.

    Returns:
        torch.Tensor: Tensor with cubic features added.
    """
    a, b = torch.triu_indices(X.shape[1], X.shape[1])
    cross_features = X[:, a] * X[:, b]            
    
    triples = [(i,j,k) for i in range(X.shape[1]) for j in range(i,X.shape[1]) for k in range(j,X.shape[1])]
    i, j, k = zip(*triples)
    i, j, k = torch.tensor(i), torch.tensor(j), torch.tensor(k)
    cubic_features = X[:, i] * X[:, j] * X[:, k]
    
    return torch.cat([X, cross_features, cubic_features], dim=1)

class ClassificationLogLikelihood(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 2, num_hidden: int = 3, hidden_dim: int = 50):
        """
        A simple feedforward neural network for classification tasks.
        Args:
            input_dim (int): Dimension of the input features.
            num_classes (int): Number of output classes.
            num_hidden (int): Number of hidden layers.
            hidden_dim (int): Number of units in each hidden layer.
        """
        super(ClassificationLogLikelihood, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_hidden):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.output_layer = nn.Linear(hidden_dim, num_classes)
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        for hidden_layer in self.hidden_layers:
            x = F.relu(hidden_layer(x))
        x = self.output_layer(x)
        return self.log_softmax(x)
    
    def negative_log_likelihood(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        log_probs = self.forward(x)
        nll = F.nll_loss(log_probs, y.long())
        return nll
    
    def entropy(self, x: torch.Tensor) -> torch.Tensor:
        log_probs = self.forward(x)
        probs = torch.exp(log_probs)
        entropy = -torch.sum(probs * log_probs, dim=1).mean()
        return entropy
    
def find_maximum_likelihood_estimates_nn_classification(X: torch.Tensor, y: torch.Tensor, num_hidden: int = 3, hidden_dim: int = 50, num_epochs: int = 10000, lr: float = 0.0001, seed: int=0):
    """
    Find the maximum likelihood estimates for a neural network classification model.
    
    Args:
        X (torch.Tensor): Input features.
        y (torch.Tensor): Target labels.
        num_hidden (int): Number of hidden layers.
        hidden_dim (int): Number of units in each hidden layer.
        num_epochs (int): Number of training epochs.
        lr (float): Learning rate for the optimizer.
        seed (int): Random seed for reproducibility.
                
    Returns:
        ClassificationLogLikelihood: Trained model.
    """
    model_input = X.reshape(X.shape[0], -1)
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = ClassificationLogLikelihood(input_dim=model_input.shape[1], num_classes=len(y.unique()), num_hidden=num_hidden, hidden_dim=hidden_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    for _ in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        loss = model.negative_log_likelihood(X, y)
        loss.backward()
        optimizer.step()
        
    return model