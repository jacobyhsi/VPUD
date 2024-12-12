import numpy as np
import scipy.optimize as opt

def binary_entropy(x: float):
    return - x*np.log2(x) - (1-x)*np.log2(1-x)

def binary_entropy_jacobian(x: float):
    return - (np.log(x) - np.log(1-x))/np.log(2)

def find_threshold_given_max_entropy_distance(y_prob: float, max_entropy_distance: float):
    """
    Finds the threshold for probabilities p(y|x,z) given the y_prob p(y|x) and maximum entropy distance, delta = |H(y|x) - H(y|x,z)|
    """
    if np.isnan(binary_entropy(y_prob)):
        raise ValueError("Entropy is nan")
    # Three cases
    # H(y|x) in [1 - delta, 1] and H(y|x) in [delta, 1 - delta] are the same
    if binary_entropy(y_prob) > max_entropy_distance:
        initial_values = np.linspace(0.1*y_prob, 0.9*y_prob, 10) if y_prob < 0.5 else np.linspace(y_prob + 0.1*(1 - y_prob), y_prob + 0.9*(1 - y_prob), 10)[::-1]
        func = lambda y: binary_entropy(y) - binary_entropy(y_prob) + max_entropy_distance
        for initial_value in initial_values:
            y_solution = opt.fsolve(func, x0=initial_value, fprime=binary_entropy_jacobian)
            
            if binary_entropy(y_solution) < binary_entropy(y_prob) - 0.5*max_entropy_distance:
                return np.abs(y_solution - y_prob)[0]

    elif binary_entropy(y_prob) == max_entropy_distance:
        return min(y_prob, 1 - y_prob)
    # H(y|x) < delta
    elif binary_entropy(y_prob) < max_entropy_distance:
        initial_values = np.linspace(y_prob, 0.5, 10) if y_prob < 0.5 else np.linspace(0.5, y_prob, 10)[::-1]
        func = lambda y: binary_entropy(y) - binary_entropy(y_prob) - max_entropy_distance
        for initial_value in initial_values:
            y_solution = opt.fsolve(func, x0=initial_value, fprime=binary_entropy_jacobian)
            
            if binary_entropy(y_solution) > binary_entropy(y_prob) + 0.5*max_entropy_distance:
                return np.abs(y_solution - y_prob)[0]
    else:
        return np.nan

def find_max_entropy_distance_given_threshold(y_prob: float, threshold: float):
    """
    Finds the maximum entropy distance, delta = |H(y|x) - H(y|x,z)| given the y_prob p(y|x) and epsilon, the threshold for probabilities p(y|x,z)
    """
    if np.isnan(binary_entropy(y_prob)):
        raise ValueError("Entropy is nan")
    
    entropy_difference = lambda epsilon: np.abs(binary_entropy(y_prob + epsilon) - binary_entropy(y_prob))
    if y_prob <= threshold:
        return max(binary_entropy(y_prob), entropy_difference(threshold))
    elif y_prob >= 1 - threshold:
        return max(binary_entropy(y_prob), entropy_difference(-threshold))
    else:
        return max(entropy_difference(threshold), entropy_difference(-threshold))