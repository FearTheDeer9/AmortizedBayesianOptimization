"""
Base environment interface for causal environments in the causal_meta library.

This module contains the abstract base class for causal environments, which
defines the interface for sampling data, performing interventions, and 
evaluating outcomes in causal systems.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Set, Optional, Union, Any, Tuple, Callable, Type
import numpy as np
import pandas as pd
import json
from numpy.random import RandomState

from causal_meta.graph.causal_graph import CausalGraph


class BaseEnvironment(ABC):
    """
    Abstract base class for causal environments.

    A causal environment represents a system where causal relationships exist
    between variables and where interventions can be performed. This class
    defines the interface for interacting with such environments, including
    methods for sampling data, performing interventions, and evaluating outcomes.

    Subclasses must implement the abstract methods to provide specific
    functionality based on the type of causal environment.
    """

    def __init__(self,
                 causal_graph: Optional[CausalGraph] = None,
                 variable_names: Optional[List[str]] = None,
                 random_state: Optional[Union[int, RandomState]] = None,
                 **kwargs):
        """
        Initialize a causal environment.

        Args:
            causal_graph: The causal graph representing the structure of the environment.
                If None, a graph must be set later or generated internally.
            variable_names: Names of the variables in the environment.
                If None, names will be generated based on the causal graph.
            random_state: Random state for reproducibility. Can be an integer seed or
                a numpy RandomState object.
            **kwargs: Additional keyword arguments for specific environment types.
        """
        self._causal_graph = causal_graph
        self._variable_names = variable_names or []

        # Initialize random state
        if isinstance(random_state, RandomState):
            self._random_state = random_state
        elif random_state is not None:
            self._random_state = RandomState(random_state)
        else:
            self._random_state = RandomState()

        # Additional environment state and configurations
        self._interventions = {}  # Current active interventions
        self._environment_config = kwargs  # Store additional configuration

    @abstractmethod
    def sample_data(self,
                    sample_size: int,
                    random_seed: Optional[int] = None) -> pd.DataFrame:
        """
        Sample data from the environment.

        This method samples data from the environment according to the current
        state, which may include active interventions.

        Args:
            sample_size: Number of samples to generate
            random_seed: Optional seed for reproducibility

        Returns:
            DataFrame: A pandas DataFrame containing the sampled data,
                with columns corresponding to variable names and rows
                representing individual samples
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        Reset the environment to its initial state.

        This method resets any active interventions and returns the environment
        to its original configuration.
        """
        pass

    @abstractmethod
    def get_observational_data(self,
                               sample_size: int,
                               random_seed: Optional[int] = None) -> pd.DataFrame:
        """
        Get purely observational data from the environment.

        This method samples observational data without any interventions,
        regardless of the current environment state.

        Args:
            sample_size: Number of samples to generate
            random_seed: Optional seed for reproducibility

        Returns:
            DataFrame: A pandas DataFrame containing the observational data
        """
        pass

    def set_random_state(self, random_state: Union[int, RandomState]) -> None:
        """
        Set the random state for the environment.

        Args:
            random_state: New random state, either as an integer seed or
                a numpy RandomState object
        """
        if isinstance(random_state, RandomState):
            self._random_state = random_state
        else:
            self._random_state = RandomState(random_state)

    def set_causal_graph(self, causal_graph: CausalGraph) -> None:
        """
        Set the causal graph for the environment.

        Args:
            causal_graph: The causal graph representing the structure

        Raises:
            ValueError: If the graph is incompatible with the environment
        """
        # Basic validation can be added here
        self._causal_graph = causal_graph

        # If variable names aren't set, generate them from the graph
        if not self._variable_names:
            self._variable_names = [str(node)
                                    for node in causal_graph.get_nodes()]

    @abstractmethod
    def do_intervention(self,
                        target: Any,
                        value: Any) -> None:
        """
        Perform a 'do' intervention on the environment.

        This sets the value of the target variable to the specified value,
        effectively cutting off the influence of its parent nodes.

        Args:
            target: The target variable to intervene on
            value: The value to set the target variable to

        Raises:
            ValueError: If the target variable doesn't exist in the environment
        """
        pass

    @abstractmethod
    def sample_interventional_data(self,
                                   interventions: Dict[Any, Any],
                                   sample_size: int,
                                   random_seed: Optional[int] = None) -> pd.DataFrame:
        """
        Sample data from the environment under specified interventions.

        This method temporarily applies the specified interventions,
        samples data, and then restores the environment to its previous state.

        Args:
            interventions: A dictionary mapping target variables to intervention values
            sample_size: Number of samples to generate
            random_seed: Optional seed for reproducibility

        Returns:
            DataFrame: A pandas DataFrame containing the interventional data

        Raises:
            ValueError: If any target variable doesn't exist in the environment
        """
        pass

    @abstractmethod
    def multiple_interventions(self,
                               interventions: Dict[Any, Any]) -> None:
        """
        Perform multiple interventions on the environment simultaneously.

        Args:
            interventions: A dictionary mapping target variables to intervention values

        Raises:
            ValueError: If any target variable doesn't exist in the environment
        """
        pass

    @abstractmethod
    def get_intervention_effects(self,
                                 target: Any,
                                 outcome: Any,
                                 intervention_values: List[Any],
                                 sample_size: int = 1000,
                                 random_seed: Optional[int] = None) -> Dict[Any, float]:
        """
        Calculate the effects of different interventions on an outcome.

        Args:
            target: The target variable to intervene on
            outcome: The outcome variable to measure
            intervention_values: List of values to intervene with
            sample_size: Number of samples to use for each intervention
            random_seed: Optional seed for reproducibility

        Returns:
            Dict[Any, float]: A dictionary mapping intervention values to outcomes

        Raises:
            ValueError: If the target or outcome variable doesn't exist
        """
        pass

    def get_active_interventions(self) -> Dict[Any, Any]:
        """
        Get the currently active interventions in the environment.

        Returns:
            Dict[Any, Any]: A dictionary of target variables to intervention values
        """
        return self._interventions.copy()

    @abstractmethod
    def evaluate_counterfactual(self,
                                factual_data: pd.DataFrame,
                                interventions: Dict[Any, Any]) -> pd.DataFrame:
        """
        Evaluate counterfactual outcomes given factual data and interventions.

        This method calculates what would have happened had the interventions
        been applied in the factual scenarios.

        Args:
            factual_data: Observed data for which to compute counterfactuals
            interventions: A dictionary mapping target variables to intervention values

        Returns:
            DataFrame: A pandas DataFrame containing the counterfactual outcomes

        Raises:
            ValueError: If any target variable doesn't exist or factual_data is invalid
        """
        pass

    @abstractmethod
    def predict_outcome(self,
                        interventions: Dict[Any, Any],
                        conditions: Optional[Dict[Any, Any]] = None) -> Any:
        """
        Predict the expected outcome of variables given interventions and conditions.

        Args:
            interventions: A dictionary mapping target variables to intervention values
            conditions: Optional conditioning values for other variables

        Returns:
            Any: The predicted outcome, typically a dictionary of variable values
                or a probability distribution

        Raises:
            ValueError: If any specified variable doesn't exist
        """
        pass

    @abstractmethod
    def compute_effect(self,
                       treatment: Any,
                       outcome: Any,
                       treatment_value: Any,
                       baseline_value: Optional[Any] = None,
                       sample_size: int = 1000,
                       random_seed: Optional[int] = None) -> float:
        """
        Compute the causal effect of a treatment on an outcome.

        This method calculates the average treatment effect (ATE) by comparing
        the outcome under treatment to the outcome under baseline.

        Args:
            treatment: The treatment variable
            outcome: The outcome variable
            treatment_value: The value to set the treatment to
            baseline_value: The baseline value (default: may depend on implementation)
            sample_size: Number of samples to use
            random_seed: Optional seed for reproducibility

        Returns:
            float: The estimated causal effect

        Raises:
            ValueError: If the treatment or outcome variable doesn't exist
        """
        pass

    @abstractmethod
    def counterfactual_distribution(self,
                                    factual_data: pd.DataFrame,
                                    interventions: Dict[Any, Any],
                                    target_variables: Optional[List[Any]] = None,
                                    num_samples: int = 100,
                                    random_seed: Optional[int] = None) -> Dict[Any, np.ndarray]:
        """
        Compute the distribution of counterfactual outcomes.

        This method estimates the distribution of outcomes in counterfactual worlds
        given factual observations and interventions.

        Args:
            factual_data: Observed data for which to compute counterfactuals
            interventions: A dictionary mapping target variables to intervention values
            target_variables: Optional list of variables for which to compute distributions
            num_samples: Number of samples to use for distribution estimation
            random_seed: Optional seed for reproducibility

        Returns:
            Dict[Any, np.ndarray]: A dictionary mapping variables to arrays of
                samples from their counterfactual distributions

        Raises:
            ValueError: If any specified variable doesn't exist
        """
        pass

    def get_causal_graph(self) -> Optional[CausalGraph]:
        """
        Get the causal graph associated with this environment.

        Returns:
            Optional[CausalGraph]: The causal graph or None if not set
        """
        return self._causal_graph

    def get_variable_names(self) -> List[str]:
        """
        Get the names of variables in the environment.

        Returns:
            List[str]: List of variable names
        """
        return self._variable_names.copy()

    def get_variable_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about the variables in the environment.

        This method returns a dictionary with variable names as keys and
        dictionaries of variable properties as values.

        Returns:
            Dict[str, Dict[str, Any]]: Dictionary of variable information
        """
        # Default implementation provides basic info,
        # subclasses should override to provide more
        var_info = {}

        if self._causal_graph:
            for var_name in self._variable_names:
                var_info[var_name] = {
                    "parents": list(self._causal_graph.get_parents(var_name)),
                    "children": list(self._causal_graph.get_children(var_name))
                }
        else:
            for var_name in self._variable_names:
                var_info[var_name] = {
                    "parents": [],
                    "children": []
                }

        return var_info

    def plot_causal_graph(self, **kwargs) -> Any:
        """
        Plot the causal graph of the environment.

        This is a convenience method that delegates to the visualization
        utilities. The actual implementation and return value depends on
        the visualization backend.

        Args:
            **kwargs: Additional arguments to pass to the visualization function

        Returns:
            Any: The plot object or None if visualization is not available

        Raises:
            ValueError: If no causal graph is set
        """
        if self._causal_graph is None:
            raise ValueError("No causal graph is set for this environment")

        # This would usually call a visualization utility
        # For now, we just return the graph itself
        return self._causal_graph

    def to_json(self) -> str:
        """
        Serialize the environment configuration to JSON.

        This method serializes the essential configuration of the environment,
        not including the current state or any runtime data.

        Returns:
            str: JSON string representation of the environment configuration

        Raises:
            NotImplementedError: If serialization is not supported by a subclass
        """
        # Basic serialization that should be extended by subclasses
        config = {
            "environment_type": self.__class__.__name__,
            "variable_names": self._variable_names,
            "config": {k: v for k, v in self._environment_config.items()
                       if isinstance(v, (str, int, float, bool, list, dict))}
        }

        return json.dumps(config)

    @classmethod
    def from_json(cls, json_str: str) -> 'BaseEnvironment':
        """
        Create an environment from a JSON configuration.

        This is a class method that creates a new environment instance
        from a JSON configuration string.

        Args:
            json_str: JSON string representation of the environment configuration

        Returns:
            BaseEnvironment: A new environment instance

        Raises:
            ValueError: If the JSON configuration is invalid
            NotImplementedError: If deserialization is not supported
        """
        # This should be implemented by subclasses to handle specific params
        raise NotImplementedError(
            "Must be implemented by concrete environment classes")

    def validate(self) -> bool:
        """
        Validate the environment configuration and state.

        This method checks if the environment is properly configured
        and in a valid state.

        Returns:
            bool: True if the environment is valid, False otherwise
        """
        # Check if a causal graph is set when required
        if hasattr(self, '_requires_graph') and self._requires_graph:
            if self._causal_graph is None:
                return False

        # Check if variable names match the causal graph nodes
        if self._causal_graph is not None:
            graph_nodes = set(str(node)
                              for node in self._causal_graph.get_nodes())
            if not set(self._variable_names).issubset(graph_nodes):
                return False

        return True

    def __str__(self) -> str:
        """
        Return a string representation of the environment.

        Returns:
            str: String representation of the environment
        """
        var_info = f"Variables: {', '.join(self._variable_names)}" if self._variable_names else "No variables defined"
        graph_info = "Causal graph set" if self._causal_graph is not None else "No causal graph set"
        interventions_info = f"Active interventions: {self._interventions}" if self._interventions else "No active interventions"

        return f"{self.__class__.__name__}({var_info}; {graph_info}; {interventions_info})"

    def __repr__(self) -> str:
        """
        Return a detailed string representation of the environment.

        Returns:
            str: Detailed string representation of the environment
        """
        return self.__str__()
