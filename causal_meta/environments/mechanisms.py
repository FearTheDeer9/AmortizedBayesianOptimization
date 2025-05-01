#!/usr/bin/env python3
"""
Causal mechanism implementations for the causal_meta library.

This module defines the base interface for causal mechanisms and provides
implementations for various types (linear, nonlinear, neural network-based).
Mechanisms define how a variable's value is computed based on its direct causes (parents).
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Union, Optional, Callable
import numpy as np
import math  # For exp, log
# Potential future imports: torch, tensorflow


class BaseMechanism(ABC):
    """
    Abstract base class for all causal mechanisms.

    A causal mechanism defines the functional relationship between a variable
    and its direct causes (parents). It specifies how to compute the variable's
    value given the values of its parents.
    """

    def __init__(self, noise_std: float = 0.0, seed: Optional[int] = None):
        """
        Initialize the base mechanism.

        Args:
            noise_std: Standard deviation of additive Gaussian noise.
                       Set to 0 for deterministic mechanisms.
            seed: Optional random seed for reproducibility of noise.
        """
        self.noise_std = noise_std
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    @abstractmethod
    def compute(self, parent_values: Dict[Any, Union[float, int]]) -> Union[float, int]:
        """
        Compute the value of the variable based on parent values.

        This method must be implemented by subclasses.

        Args:
            parent_values: A dictionary mapping parent node IDs to their values.

        Returns:
            The computed value of the variable.
        """
        pass

    def __call__(self, parent_values: Dict[Any, Union[float, int]]) -> Union[float, int]:
        """
        Compute the variable's value, potentially adding noise.

        Handles input validation and calls the specific compute logic, then adds
        noise if specified.

        Args:
            parent_values: A dictionary mapping parent node IDs to their values.

        Returns:
            The computed value, possibly with noise added.
        """
        # Input validation can be added here if needed
        computed_value = self.compute(parent_values)

        # Add noise if specified
        if self.noise_std > 0:
            noise = self._rng.normal(loc=0.0, scale=self.noise_std)
            # Ensure type consistency if necessary (e.g., casting to int if variable is discrete)
            # Note: This assumes additive Gaussian noise. Subclasses might override
            # this __call__ method or the noise generation process if needed.
            if isinstance(computed_value, int):
                # If the deterministic part is an integer, maybe we want integer output?
                # This is a design choice - for now, keep it float if noise is added.
                computed_value = float(computed_value) + noise
            else:
                computed_value += noise

        return computed_value

    def set_seed(self, seed: int) -> None:
        """Resets the random number generator with a new seed."""
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def __repr__(self) -> str:
        """Return a string representation of the mechanism."""
        return f"{self.__class__.__name__}(noise_std={self.noise_std}, seed={self.seed})"

    @property
    @abstractmethod
    def expected_num_parents(self) -> int:
        """Number of parent variables expected by this mechanism."""
        pass

# --- Concrete implementations will follow below ---
# e.g., LinearMechanism, NonlinearMechanism, NeuralNetworkMechanism


class LinearMechanism(BaseMechanism):
    """
    Implements a linear causal mechanism.

    The variable's value is computed as a weighted sum of its parents' values,
    plus an optional intercept and additive noise.
    Equation: Y = intercept + sum(weight_i * parent_i) + noise
    """

    def __init__(self,
                 weights: Dict[Any, float],
                 intercept: float = 0.0,
                 noise_std: float = 0.0,
                 seed: Optional[int] = None):
        """
        Initialize the linear mechanism.

        Args:
            weights: A dictionary mapping parent node IDs to their corresponding
                     linear coefficients (weights).
            intercept: The intercept term (bias) for the linear equation. Default is 0.
            noise_std: Standard deviation of additive Gaussian noise. Default is 0.
            seed: Optional random seed for reproducibility of noise.
        """
        super().__init__(noise_std=noise_std, seed=seed)
        self.weights = weights
        self.intercept = intercept
        # Store expected parent keys for validation during compute
        self._expected_parents = set(weights.keys())

    def compute(self, parent_values: Dict[Any, Union[float, int]]) -> float:
        """
        Compute the value using the linear equation.

        Args:
            parent_values: A dictionary mapping parent node IDs to their values.
                           Must contain exactly the parents specified in `weights`.

        Returns:
            The computed value as a float.

        Raises:
            ValueError: If the keys in parent_values do not match the expected parents
                        defined by the weights during initialization.
        """
        if set(parent_values.keys()) != self._expected_parents:
            raise ValueError(f"Input parents {set(parent_values.keys())} do not match "
                             f"expected parents {self._expected_parents}")

        # Calculate weighted sum
        value = self.intercept
        for parent_id, weight in self.weights.items():
            value += weight * parent_values[parent_id]

        # Ensure float output even if all inputs/weights are int
        return float(value)

    def __repr__(self) -> str:
        """Return a string representation of the linear mechanism."""
        # Shorten weights dict for cleaner repr if it's too long
        weights_repr = f"{len(self.weights)} parents" if len(
            self.weights) > 3 else str(self.weights)
        return (f"{self.__class__.__name__}("
                f"weights={weights_repr}, "
                f"intercept={self.intercept}, "
                f"noise_std={self.noise_std}, "
                f"seed={self.seed})")

    @property
    def expected_num_parents(self) -> int:
        return len(self.weights)

    @classmethod
    def generate_random_weights(
        cls,
        num_parents: int,
        loc: float = 0.0,
        scale: float = 1.0,
        random_state: Optional[np.random.RandomState] = None
    ) -> Dict[Any, float]:
        """Generates random weights for a linear mechanism.

        Args:
            num_parents (int): The number of parent variables.
            loc (float): The mean of the distribution for weights. Defaults to 0.0.
            scale (float): The standard deviation of the distribution for weights. Defaults to 1.0.
            random_state (Optional[np.random.RandomState]): Random state for generation.

        Returns:
            Dict[Any, float]: Dictionary mapping parent node IDs to their corresponding weights.
        """
        rs = random_state if random_state is not None else np.random.RandomState()
        return {i: rs.normal(loc, scale) for i in range(num_parents)}


class NonlinearMechanism(BaseMechanism):
    """
    Implements a nonlinear causal mechanism using predefined or custom functions.

    The variable's value is computed by applying a nonlinear function to a
    weighted sum of its parents' values (similar to LinearMechanism input),
    plus optional noise.
    Equation: Y = func(intercept + sum(weight_i * parent_i)) + noise
    Alternatively, if a custom function is provided, it might operate directly
    on the parent_values dictionary.
    """

    def __init__(self,
                 func_type: Union[str, Callable],
                 # Optional if func takes dict
                 weights: Optional[Dict[Any, float]] = None,
                 intercept: float = 0.0,
                 noise_std: float = 0.0,
                 # Params for func, e.g., degree for poly
                 func_params: Optional[Dict[str, Any]] = None,
                 seed: Optional[int] = None):
        """
        Initialize the nonlinear mechanism.

        Args:
            func_type: Specifies the nonlinear function. Can be a string identifier
                       (e.g., 'quadratic', 'sigmoid', 'exp', 'log', 'custom') or a
                       callable function directly. If 'custom', the callable must
                       be provided in func_params['custom_func'].
            weights: A dictionary mapping parent node IDs to their weights for the
                     linear combination input to the function. Required unless
                     func_type is a custom callable that expects the raw parent_values dict.
            intercept: The intercept term added before applying the nonlinear function.
            noise_std: Standard deviation of additive Gaussian noise.
            func_params: Optional dictionary of parameters for the function.
                         - For 'quadratic': {'degree': 2 (default), 'coeff': 1.0 (default)}
                         - For 'custom': {'custom_func': Callable} -> The function must
                           accept a dictionary parent_values as input.
                         - Other funcs might define their own params (e.g., base for log).
            seed: Optional random seed for reproducibility of noise.
        """
        super().__init__(noise_std=noise_std, seed=seed)

        if isinstance(func_type, str) and func_type.lower() == 'custom':
            if not func_params or 'custom_func' not in func_params or not callable(func_params['custom_func']):
                raise ValueError(
                    "For 'custom' func_type, 'func_params' must contain a callable 'custom_func'")
            self.func = func_params['custom_func']
            # Weights might not be needed if the custom func handles parents directly
            if weights is None:
                weights = {}  # Assign empty dict if none provided for custom func
            self._is_custom_dict_func = True  # Flag that func takes dict directly
        elif callable(func_type):
            self.func = func_type
            # Assume callable func_type takes linear combination unless weights is explicitly None
            self._is_custom_dict_func = (weights is None)
            if self._is_custom_dict_func:
                weights = {}
        elif isinstance(func_type, str):
            self.func = self._get_predefined_func(
                func_type.lower(), func_params)
            self._is_custom_dict_func = False  # Predefined funcs take linear combination
            if weights is None:
                raise ValueError(
                    f"Weights dictionary required for predefined func_type '{func_type}'")
        else:
            raise TypeError(
                f"Unsupported func_type: {type(func_type)}. Must be str or callable.")

        self.func_type_name = func_type if isinstance(
            func_type, str) else func_type.__name__
        self.weights = weights if weights is not None else {}
        self.intercept = intercept
        self.func_params = func_params if func_params else {}
        self._expected_parents = set(self.weights.keys())

    def _get_predefined_func(self, name: str, params: Optional[Dict]) -> Callable:
        """Maps string names to predefined nonlinear functions."""
        params = params if params else {}
        if name == 'quadratic':
            degree = params.get('degree', 2)
            coeff = params.get('coeff', 1.0)
            if degree != 2:
                # Or raise error
                print(
                    "Warning: 'quadratic' func_type used, but degree specified is not 2.")
            return lambda x: coeff * (x ** 2)
        elif name == 'polynomial':  # More general polynomial
            degree = params.get('degree', 2)
            # Example: [c0, c1, c2] for degree 2
            coeffs = params.get('coeffs', [1.0] * (degree + 1))
            if len(coeffs) != degree + 1:
                raise ValueError("Length of coeffs must be degree + 1")
            # Note: Assumes coeffs are [intercept, linear, quadratic, ...] applied to the *result* of linear combination
            # This might need refinement depending on desired behavior. A simpler poly:
            return lambda x: coeffs[0] + sum(c * (x ** i) for i, c in enumerate(coeffs[1:], 1))
            # Or just a simple power function for now:
            # return lambda x: x ** degree
        elif name == 'sigmoid':
            # Logistic sigmoid: 1 / (1 + exp(-x))
            # Avoid overflow
            return lambda x: 1 / (1 + math.exp(-x)) if -x < 700 else 0
        elif name == 'exp':
            # Avoid overflow
            return lambda x: math.exp(x) if x < 700 else float('inf')
        elif name == 'log':
            base = params.get('base', math.e)
            # Add small epsilon for stability if input can be <= 0
            epsilon = params.get('epsilon', 1e-9)
            return lambda x: math.log(x + epsilon, base) if x + epsilon > 0 else -float('inf')
        # Add more predefined functions here (e.g., tanh, relu)
        else:
            raise ValueError(f"Unknown predefined function type: {name}")

    def compute(self, parent_values: Dict[Any, Union[float, int]]) -> float:
        """
        Compute the value using the nonlinear equation.

        Args:
            parent_values: A dictionary mapping parent node IDs to their values.

        Returns:
            The computed value as a float.

        Raises:
            ValueError: If required parent values are missing for the linear combination.
        """
        if self._is_custom_dict_func:
            # Custom function handles parent values directly
            # We might still want validation based on expected keys if provided initially
            if self._expected_parents and set(parent_values.keys()) != self._expected_parents:
                print(f"Warning: Custom function input parents {set(parent_values.keys())} "
                      f"don't match expected {self._expected_parents}")
                # Depending on function, might still work or raise error internally
            try:
                # Pass extra args if needed
                result = self.func(
                    parent_values, **self.func_params.get('custom_args', {}))
            except TypeError:  # Try without extra args if func doesn't take them
                result = self.func(parent_values)
            return float(result)
        else:
            # Predefined function takes the linear combination
            if set(parent_values.keys()) != self._expected_parents:
                raise ValueError(f"Input parents {set(parent_values.keys())} do not match "
                                 f"expected parents {self._expected_parents} for linear combination")

            # Calculate linear combination
            linear_combination = self.intercept
            for parent_id, weight in self.weights.items():
                linear_combination += weight * parent_values[parent_id]

            # Apply the nonlinear function
            result = self.func(linear_combination)
            return float(result)  # Ensure float output

    def __repr__(self) -> str:
        """Return a string representation of the nonlinear mechanism."""
        weights_repr = f"{len(self.weights)} parents" if len(
            self.weights) > 3 else str(self.weights)
        func_repr = self.func_type_name
        params_repr = f", func_params={self.func_params}" if self.func_params else ""

        # Avoid showing weights if custom func takes dict directly and no weights were specified
        weights_intercept_repr = ""
        if not (self._is_custom_dict_func and not self.weights):
            weights_intercept_repr = f"weights={weights_repr}, intercept={self.intercept}, "

        return (f"{self.__class__.__name__}("
                f"func_type={func_repr}, "
                f"{weights_intercept_repr}"
                f"noise_std={self.noise_std}{params_repr}, "
                f"seed={self.seed})")

    @property
    def expected_num_parents(self) -> int:
        return len(self.weights)
