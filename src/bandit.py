import numpy as np
import pandas as pd



class MultiArmBandit:
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
    
    def get_reward(self, action: int|str, **kwargs) -> float|int:
        """
        Get the reward for a given action in a given context.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def get_optimal_mean_reward(self, **kwargs) -> float|int:
        """
        Get the optimal mean reward for a given action in a given context.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def get_context_feature_cols(self) -> dict:
        """
        Get the context features for the bandit.
        """
        return []
    
    def get_action_space(self) -> list:
        """
        Get the action space for the bandit.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def get_next_context(self) -> pd.DataFrame:
        """
        Get the context for the bandit.
        """
        return None
    
    def optimal_action(self, **kwargs) -> int:
        """
        Get the best action for the bandit.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
    
class ClassificationBandit(MultiArmBandit):
    def get_reward_space(self) -> list:
        """
        Get the reward values for the bandit.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
    
class RegressionBandit(MultiArmBandit):
    pass   
class ButtonsBandit(ClassificationBandit):
    def __init__(self, num_arms: int = 2, midpoint: float = 0.5, gap: float = 0.2, seed: int = 0, **kwargs):
        """
        Initialize the bandit with a number of buttons.
        """
        super().__init__(seed)
        self.num_arms = num_arms
        self.best_arm = self.rng.integers(0, num_arms)
        self.midpoint = midpoint
        self.gap = gap
    
    def get_reward(self, action: int|str) -> float|int:
        """
        Get the reward for a given button action.
        """
        
        if isinstance(action, str):
            action = int(action)
        
        if action not in range(self.num_arms):
            raise ValueError(f"Action {action} is not a valid arm.")
        
        if action == self.best_arm:
            reward = self.rng.binomial(1, self.midpoint + self.gap*0.5)
        else:
            reward = self.rng.binomial(1, self.midpoint - self.gap*0.5)

        return reward
    
    def get_optimal_mean_reward(self) -> float|int:
        """
        Get the optimal mean reward for a given button action.
        """
        
        return self.midpoint + self.gap*0.5
    
    def get_action_space(self) -> list:
        """
        Get the action space for the bandit.
        """
        
        return list(range(self.num_arms))
    
    def get_reward_space(self):
        return ["0", "1"]
    
    def optimal_action(self, **kwargs) -> int:
        return self.best_arm
    
class SimpleContextBandit(ClassificationBandit):
    def __init__(self, num_arms: int = 2, seed: int = 0, **kwargs):
        """
        Bandit with 1d context x in [0,1] and two actions.
        action 0 passes (reward = 0) and action 1 "risks" (reward = 1 with p=x, otherwise reward = -1).
        action 0 is optimal for context < 0.5, action 1 is optimal for context > 0.5.
        """
        super().__init__(seed)
        self.num_arms = num_arms
        self.pass_arm = self.rng.integers(0, num_arms)
        self.risk_arm = 1 - self.pass_arm
        self.x = None
        self.y = None
    
    def get_context_feature_cols(self) -> list:
        """
        Get the context features for the bandit.
        """
        return ["x", "y"]
    
    def get_next_context(self) -> dict:
        """
        Get the context for the bandit.
        """
        self.x = np.round(self.rng.uniform(0, 1),3)
        self.y = np.round(self.rng.uniform(0, 1),3)
        return {"x": self.x, "y": self.y}
    
    def get_reward(self, action: int|str) -> float|int: 
        """
        Get the reward for a given action in a given context.
        """
        
        if isinstance(action, str):
            action = int(action)
        
        if action not in range(self.num_arms):
            raise ValueError(f"Action {action} is not a valid arm.")
        
        if action == self.pass_arm:
            reward = 1
        else:
            reward = 1+int(2*(self.rng.binomial(1, self.x) - 0.5))

        return reward
    
    def get_optimal_mean_reward(self) -> float|int:
        """
        Get the optimal mean reward for a given button action.
        """
        
        return 1 if self.x<0.5 else 2*(self.x-0.5)+1

    def get_mean_reward(self, action: int|str) -> float|int:
        """
        Get the mean reward for a given action in a given context.
        """
        
        if isinstance(action, str):
            action = int(action)
        
        if action not in range(self.num_arms):
            raise ValueError(f"Action {action} is not a valid arm.")
        
        if action == self.pass_arm:
            return 1
        else:
            return 2*(self.x-0.5)+1
    
    def get_action_space(self) -> list:
        """
        Get the action space for the bandit.
        """
        
        return list(range(self.num_arms))
    
    def get_reward_space(self):
        return ["0", "1", "2"]
    
    def optimal_action(self, **kwargs) -> int:
        return self.pass_arm if self.x<0.5 else self.risk_arm

class WheelBandit(ClassificationBandit):
    def __init__(self, delta = 0.5, seed: int = 0, **kwargs):

        super().__init__(seed)
        self.k = 5  # Number of actions
        self.delta = delta
        # Reward means
        self.r1 = 1  # Constant arm (a1)
        self.r2 = 0  # Suboptimal arms
        self.r3 = 5 # Optimal arms in outer regions

        self.a0, self.a1, self.a2, self.a3, self.a4 = tuple(self.rng.permutation([0, 1, 2, 3, 4]))

        self.x = self.rng.uniform(-1, 1)
        self.y = self.rng.uniform(-1, 1)
    
    def get_context_feature_cols(self) -> list:
        """
        Get the context features for the bandit.
        """
        return ["x", "y"]
    
    def get_next_context(self) -> dict:
        """
        Get the context for the bandit.
        """
        self.x = self.rng.uniform(-1, 1)
        self.y = self.rng.uniform(-1, 1)
        while self.x**2 + self.y**2 > 1:
            self.x = self.rng.uniform(-1, 1)
            self.y = self.rng.uniform(-1, 1)

        return {"x": self.x, "y": self.y}
    
    def get_mean_reward(self, action: int|str) -> float|int: 
        """
        Get the reward for a given action in a given context.
        """
        norm = np.sqrt(self.x**2 + self.y**2)
        optimal = self.optimal_action()

        if action == self.a0:
            reward = self.r1
        elif norm <= self.delta:
            reward = self.r2
        elif action == optimal:
            reward = self.r3
        else:
            reward = self.r2

        return reward

    def get_reward(self, action: int|str) -> float|int:
        return self.get_mean_reward(action)
    
    def get_optimal_mean_reward(self) -> float|int:
        """
        Get the optimal mean reward for a given button action.
        """
        norm = np.sqrt(self.x**2 + self.y**2)
        if norm<=self.delta:
            return self.r1
        else:
            return self.r3
    
    def get_action_space(self) -> list:
        """
        Get the action space for the bandit.
        """
        
        return list(range(self.k))
    
    def get_reward_space(self):
        return [str(self.r1), str(self.r2), str(self.r3)]
    
    def optimal_action(self, **kwargs) -> int:
        x, y = self.x, self.y

        if self.x**2 + self.y**2 <= self.delta:
            return self.a0  # Only a1 is optimal in inner circle
        if x > 0 and y > 0:
            return self.a1
        elif x > 0 and y < 0:
            return self.a2
        elif x < 0 and y < 0:
            return self.a3
        elif x < 0 and y > 0:
            return self.a4
        else:
            raise NotImplementedError("Bug")
    
class ButtonsRegressionBandit(RegressionBandit):
    def __init__(self, num_arms: int = 2, gap: float = 0.2, midpoint: float = 0.5, seed: int = 0, noise: float = 0.4, **kwargs):
        """
        Initialize the bandit with a number of buttons.
        """
        super().__init__(seed)
        self.num_arms = num_arms
        self.best_arm = self.rng.integers(0, num_arms)
        self.gap = gap
        self.midpoint = midpoint
        self.noise = noise
    
    def get_reward(self, action: int|str) -> float|int:
        """
        Get the reward for a given button action.
        """
        
        if isinstance(action, str):
            action = int(action)
        
        if action not in range(self.num_arms):
            raise ValueError(f"Action {action} is not a valid arm.")
        
        if action == self.best_arm:
            reward = self.rng.normal(self.midpoint + self.gap/2, self.noise)
        else:
            reward = self.rng.normal(self.midpoint - self.gap/2, self.noise)

        return np.round(reward,2)
    
    def get_optimal_mean_reward(self) -> float|int:
        """
        Get the optimal mean reward for a given button action.
        """
        
        return self.midpoint + self.gap
    
    def get_action_space(self) -> list:
        """
        Get the action space for the bandit.
        """
        
        return list(range(self.num_arms))
    
    def optimal_action(self, **kwargs) -> int:
        return self.best_arm    

BANDIT_TYPE_TO_CLASS = {
    "buttons": ButtonsBandit,
    "simple_context": SimpleContextBandit,
    "wheel": WheelBandit,
    "buttons_regression": ButtonsRegressionBandit,
}

def get_bandit(
    bandit_name: str = "buttons",
    **bandit_kwargs
) -> MultiArmBandit:
    """
    Get a bandit instance based on the bandit type.
    """
    if bandit_name not in BANDIT_TYPE_TO_CLASS:
        raise ValueError(f"Bandit type {bandit_name} is not supported.")
    
    bandit_class = BANDIT_TYPE_TO_CLASS[bandit_name]
    return bandit_class(**bandit_kwargs)