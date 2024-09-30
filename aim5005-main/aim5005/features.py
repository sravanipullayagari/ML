import numpy as np
from typing import List, Tuple
### YOU MANY NOT ADD ANY MORE IMPORTS (you may add more typing imports)

class MinMaxScaler:
    def __init__(self):
        self.minimum = None
        self.maximum = None
        
    def _check_is_array(self, x:np.ndarray) -> np.ndarray:
        """
        Try to convert x to a np.ndarray if it'a not a np.ndarray and return. If it can't be cast raise an error
        """
        if not isinstance(x, np.ndarray):
            x = np.array(x)
            
        assert isinstance(x, np.ndarray), "Expected the input to be a list"
        return x
        
    
    def fit(self, x:np.ndarray) -> None:   
        x = self._check_is_array(x)
        self.minimum=x.min(axis=0)
        self.maximum=x.max(axis=0)
        
    def transform(self, x:np.ndarray) -> list:
        """
        MinMax Scale the given vector
        """
        x = self._check_is_array(x)
        diff_max_min = self.maximum - self.minimum
        diff_max_min[diff_max_min==0] =1
        
        return (x-self.minimum) /(diff_max_min)
    
    def fit_transform(self, x:list) -> np.ndarray:
        x = self._check_is_array(x)
        self.fit(x)
        return self.transform(x)
    

class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def _check_is_array(self, x: np.ndarray) -> np.ndarray:
        """
        Try to convert x to a np.ndarray if it's not a np.ndarray and return. If it can't be cast raise an error
        """
        if not isinstance(x, np.ndarray):
            x = np.array(x)
            
        assert isinstance(x, np.ndarray), "Expected the input to be a list"
        return x

    def fit(self, x: np.ndarray) -> None:
        x = self._check_is_array(x)
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0)
        self.std[self.std == 0] = 1 

    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        Standardize features by removing the mean and scaling to unit variance.
        """
        x = self._check_is_array(x)
        return (x - self.mean) / self.std

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        x = self._check_is_array(x)
        self.fit(x)
        return self.transform(x)

class LabelEncoder:
    def __init__(self):
        self.classes_ = None

    def fit(self, y):
        y = np.array(y)
        self.classes_ = np.unique(y)
        return self

    def transform(self, y):
        if self.classes_ is None:
            raise ValueError("LabelEncoder has not been fitted yet.")
        y = np.array(y)
        encoded = np.searchsorted(self.classes_, y)
        
        # Check if there are any labels that weren't in the classes_ (unseen labels)
        if not np.all(np.isin(y, self.classes_)):
            raise ValueError("y contains new labels that weren't seen in fit().")
        return encoded

    def fit_transform(self, y):
        self.fit(y)
        return self.transform(y)


  
