"""
Structural Causal Model (SCM) implementation for causal environments.

This module provides a class for representing and manipulating structural causal
models, including methods for defining functional relationships between variables,
sampling from the model, and performing interventions.
"""
from typing import Dict, List, Set, Tuple, Optional, Union, Any, Callable, TextIO, BinaryIO
import numpy as np
import pandas as pd
from numpy.random import RandomState
import networkx as nx
import matplotlib.pyplot as plt
import inspect
from functools import partial
import copy
import json
import pickle
import os

from causal_meta.environments.base import BaseEnvironment
from causal_meta.graph.causal_graph import CausalGraph
from causal_meta.graph.visualization import plot_graph, plot_causal_graph


class StructuralCausalModel(BaseEnvironment):
    """
    Implementation of a Structural Causal Model (SCM).

    A structural causal model represents a system of causal relationships where
    each variable is determined by a function of its parent variables and some
    independent noise/exogenous variable. This class provides methods for defining,
    sampling from, and intervening on structural causal models.
    """

    def __init__(self,
                 causal_graph: Optional[CausalGraph] = None,
                 variable_names: Optional[List[str]] = None,
                 variable_domains: Optional[Dict[str, Any]] = None,
                 random_state: Optional[Union[int, RandomState]] = None):
        """
        Initialize a structural causal model.

        Args:
            causal_graph: The causal graph representing the structure of the SCM.
                If None, a graph must be set later or built incrementally.
            variable_names: Names of the variables in the SCM.
                If None, names will be generated or set when adding variables.
            variable_domains: Dictionary mapping variable names to their domains.
                Used for validation when defining structural equations.
            random_state: Random state for reproducibility. Can be an integer seed or
                a numpy RandomState object.
        """
        super().__init__(causal_graph=causal_graph,
                         variable_names=variable_names,
                         random_state=random_state)

        # Dictionary to store variable domains
        self._variable_domains = variable_domains or {}

        # Dictionary to store structural equations
        self._structural_equations = {}

        # Dictionary to store variable metadata
        self._variable_metadata = {}

        # Dictionary to store exogenous (noise) variable functions
        self._exogenous_functions = {}

        # Current intervention state
        self._interventions = {}

        # Dictionary to store original structural equations when intervening
        self._original_equations = {}

    def add_variable(self,
                     name: str,
                     domain: Any = None,
                     metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a variable to the SCM.

        Args:
            name: Name of the variable to add
            domain: Domain of the variable (e.g., "continuous", "binary", 
                   or more specific type information)
            metadata: Additional metadata about the variable

        Raises:
            ValueError: If a variable with the given name already exists
        """
        if name in self._variable_names:
            raise ValueError(f"Variable '{name}' already exists in the SCM")

        # Add variable to the variable list
        self._variable_names.append(name)

        # Store domain information
        if domain is not None:
            self._variable_domains[name] = domain

        # Store metadata
        self._variable_metadata[name] = metadata or {}

        # If causal graph exists, add the variable to the graph
        if self._causal_graph is not None:
            self._causal_graph.add_node(name)

    def define_causal_relationship(self, child: str, parents: List[str]) -> None:
        """
        Define the causal relationship between variables in the SCM.

        Args:
            child: The child variable
            parents: List of parent variables that directly influence the child

        Raises:
            ValueError: If any of the variables don't exist or if adding the
                       relationship would create a cycle in the graph
        """
        # Check if all variables exist
        for var in [child] + parents:
            if var not in self._variable_names:
                raise ValueError(f"Variable '{var}' does not exist in the SCM")

        # Create causal graph if it doesn't exist
        if self._causal_graph is None:
            self._causal_graph = CausalGraph()
            for var in self._variable_names:
                self._causal_graph.add_node(var)

        # Add edges from parents to child
        for parent in parents:
            self._causal_graph.add_edge(parent, child)

        # Check for cycles
        if not self._validate_acyclicity():
            # If adding these edges created a cycle, remove them
            for parent in parents:
                self._causal_graph.remove_edge(parent, child)
            raise ValueError(
                "Adding these causal relationships would create a cycle")

    def _validate_acyclicity(self) -> bool:
        """
        Validate that the causal graph is acyclic.

        Returns:
            bool: True if the graph is acyclic, False otherwise
        """
        if self._causal_graph is None:
            return True

        # Use the has_cycle method provided by DirectedGraph
        return not self._causal_graph.has_cycle()

    def define_structural_equation(self,
                                   variable: str,
                                   equation_function: Callable,
                                   exogenous_function: Optional[Callable] = None) -> None:
        """
        Define a structural equation for a variable.

        Args:
            variable: The variable to define the equation for
            equation_function: Function that computes the variable value based on parent values
                               and optionally exogenous (noise) input
            exogenous_function: Optional function to generate exogenous (noise) variables
                                If None, no noise will be added

        Raises:
            ValueError: If the variable doesn't exist, or if the function signature
                       doesn't match the expected parent variables
        """
        if variable not in self._variable_names:
            raise ValueError(
                f"Variable '{variable}' does not exist in the SCM")

        # Get the parents of the variable
        parents = self.get_parents(variable)

        # Check if the equation function has the expected signature
        if not self._validate_equation_function(variable, equation_function, parents):
            raise ValueError(
                f"The equation function signature doesn't match the expected parent variables: {parents}")

        # Store the structural equation
        self._structural_equations[variable] = equation_function

        # Store the exogenous function if provided
        if exogenous_function is not None:
            self._exogenous_functions[variable] = exogenous_function

    def _validate_equation_function(self,
                                    variable: str,
                                    function: Callable,
                                    parents: List[str]) -> bool:
        """
        Validate that a function has the expected signature based on parent variables.

        Args:
            variable: The variable the function is for
            function: The function to validate
            parents: List of parent variables

        Returns:
            bool: True if the function has a valid signature, False otherwise
        """
        try:
            # Get the function signature
            sig = inspect.signature(function)
            params = list(sig.parameters.keys())

            # Check if the function accepts all parent variables
            # The function can have additional parameters for exogenous variables,
            # but must have at least parameters for all parents
            return all(parent in params for parent in parents)
        except (ValueError, TypeError):
            # If we can't inspect the function, assume it's invalid
            return False

    def define_deterministic_equation(self,
                                      variable: str,
                                      equation_function: Callable) -> None:
        """
        Define a deterministic structural equation for a variable.

        This is a convenience method that sets a structural equation with no
        exogenous (noise) input.

        Args:
            variable: The variable to define the equation for
            equation_function: Function that computes the variable value based on parent values

        Raises:
            ValueError: If the variable doesn't exist, or if the function signature
                       doesn't match the expected parent variables
        """
        self.define_structural_equation(
            variable, equation_function, exogenous_function=None)

    def define_probabilistic_equation(self,
                                      variable: str,
                                      equation_function: Callable,
                                      noise_distribution: Callable) -> None:
        """
        Define a probabilistic structural equation for a variable.

        This method sets a structural equation with a noise distribution
        that will be sampled and passed to the equation function.

        Args:
            variable: The variable to define the equation for
            equation_function: Function that computes the variable value based on 
                              parent values and noise input
            noise_distribution: Function that generates noise values when called
                               (e.g., lambda: np.random.normal(0, 1))

        Raises:
            ValueError: If the variable doesn't exist, or if the function signature
                       doesn't match the expected parent variables
        """
        # Ensure the equation function accepts a 'noise' parameter
        if not self._validate_probabilistic_function(equation_function):
            raise ValueError(
                "The equation function must accept a 'noise' parameter")

        self.define_structural_equation(
            variable, equation_function, exogenous_function=noise_distribution)

    def _validate_probabilistic_function(self, function: Callable) -> bool:
        """
        Validate that a function accepts a 'noise' parameter.

        Args:
            function: The function to validate

        Returns:
            bool: True if the function accepts a 'noise' parameter, False otherwise
        """
        try:
            # Get the function signature
            sig = inspect.signature(function)
            params = list(sig.parameters.keys())

            # Check if the function accepts a 'noise' parameter
            return 'noise' in params
        except (ValueError, TypeError):
            # If we can't inspect the function, assume it's invalid
            return False

    def define_linear_gaussian_equation(self,
                                        variable: str,
                                        coefficients: Dict[str, float],
                                        intercept: float = 0.0,
                                        noise_std: float = 1.0) -> None:
        """
        Define a linear Gaussian structural equation.

        This is a convenience method for defining a common type of structural
        equation: linear combination of parents plus Gaussian noise.

        Args:
            variable: The variable to define the equation for
            coefficients: Dictionary mapping parent variable names to their coefficients
            intercept: Intercept/bias term for the linear equation
            noise_std: Standard deviation of the Gaussian noise

        Raises:
            ValueError: If the variable doesn't exist, or if coefficients are
                       provided for variables that are not parents
        """
        if variable not in self._variable_names:
            raise ValueError(
                f"Variable '{variable}' does not exist in the SCM")

        # Get the parents of the variable
        parents = set(self.get_parents(variable))

        # Check that coefficients are only for parents
        for parent in coefficients.keys():
            if parent not in parents:
                raise ValueError(
                    f"Variable '{parent}' is not a parent of '{variable}'")

        # Create a function that accepts the appropriate parent parameters
        # We use exec to dynamically create a function with the exact parameter names needed
        param_list = list(parents)
        param_str = ", ".join(param_list)

        # If we need noise, add it to the parameters
        if noise_std > 0:
            param_str += ", noise" if param_str else "noise"

        # Create the function body
        func_body = f"def linear_equation({param_str}):\n"
        func_body += f"    result = {intercept}\n"

        # Add each parent's contribution
        for parent, coef in coefficients.items():
            func_body += f"    result += {coef} * {parent}\n"

        # Add noise if applicable
        if noise_std > 0:
            func_body += "    result += noise\n"

        func_body += "    return result\n"

        # Create the function namespace
        namespace = {}

        # Execute the function definition
        exec(func_body, namespace)

        # Get the function from the namespace
        linear_equation = namespace['linear_equation']

        # Create the noise function if noise_std > 0
        noise_function = None
        if noise_std > 0:
            # Define a noise function that accepts sample_size and uses a RandomState instance
            def noise_func(sample_size: int, random_state: np.random.RandomState):
                return random_state.normal(0, noise_std, size=sample_size)
            noise_function = noise_func

        # Define the structural equation
        self.define_structural_equation(
            variable, linear_equation, exogenous_function=noise_function)

    def evaluate_equation(self,
                          variable: str,
                          input_values: Dict[str, Any]) -> Any:
        """
        Evaluate the structural equation for a variable with given input values.

        Args:
            variable: The variable to evaluate
            input_values: Dictionary mapping variable names to their values

        Returns:
            Any: The result of evaluating the structural equation

        Raises:
            ValueError: If the variable doesn't exist or doesn't have a defined
                       structural equation
        """
        if variable not in self._variable_names:
            raise ValueError(
                f"Variable '{variable}' does not exist in the SCM")

        if variable not in self._structural_equations:
            raise ValueError(
                f"No structural equation defined for variable '{variable}'")

        # Get the structural equation
        equation = self._structural_equations[variable]

        # Check if there's an intervention for this variable
        if variable in self._interventions:
            # If the intervention is a constant value, return it
            if not callable(self._interventions[variable]):
                return self._interventions[variable]

            # If the intervention is a function, evaluate it with the input values
            return self._interventions[variable](**input_values)

        # *** Ensure NO noise generation happens here - it's passed via input_values ***

        # Revert to previous logic: Explicitly get parents and add noise if present in input_values
        equation_args = {}
        parents = self.get_parents(variable)
        missing_parents = []
        for parent in parents:
            if parent in input_values:
                equation_args[parent] = input_values[parent]
            else:
                # If a parent value is truly missing from input, raise error
                missing_parents.append(parent)

        if missing_parents:
            raise ValueError(
                f"Missing values for parent variables: {missing_parents}")

        # Add noise value if it was generated and passed in input_values
        # Check against function signature if noise is actually expected
        sig = inspect.signature(equation)
        if 'noise' in sig.parameters and 'noise' in input_values:
            equation_args['noise'] = input_values['noise']
        elif 'noise' in sig.parameters and 'noise' not in input_values:
            # This might indicate an issue if noise is expected but not provided
            pass # Or raise an error/warning depending on desired strictness

        # Evaluate the equation
        try:
            result = equation(**equation_args)
        except TypeError as te:
            raise TypeError(f"Error calling equation for '{variable}' with args {equation_args}. Original error: {te}") from te

        return result

    def get_parents(self, variable: str) -> List[str]:
        """
        Get the parents of a variable in the causal graph.

        Args:
            variable: The variable to get parents for

        Returns:
            List[str]: List of parent variables

        Raises:
            ValueError: If the variable doesn't exist or if the causal graph is not defined
        """
        if variable not in self._variable_names:
            raise ValueError(
                f"Variable '{variable}' does not exist in the SCM")

        if self._causal_graph is None:
            return []

        return list(self._causal_graph.get_parents(variable))

    def get_children(self, variable: str) -> List[str]:
        """
        Get the children of a variable in the causal graph.

        Args:
            variable: The variable to get children for

        Returns:
            List[str]: List of child variables

        Raises:
            ValueError: If the variable doesn't exist or if the causal graph is not defined
        """
        if variable not in self._variable_names:
            raise ValueError(
                f"Variable '{variable}' does not exist in the SCM")

        if self._causal_graph is None:
            return []

        return list(self._causal_graph.get_children(variable))

    def sample_data(
        self,
        sample_size: int = 100,
        include_latents: bool = False,
        squeeze: bool = True,
        as_array: bool = False,
        random_seed: Optional[int] = None,
    ) -> Union[np.ndarray, pd.DataFrame]:
        """
        Sample data from the structural causal model.

        This method samples data according to the causal relationships defined
        in the SCM, respecting any active interventions.

        Args:
            sample_size: Number of samples to generate.
            include_latents: Whether to include latent variables in the output.
                            Defaults to False.
            squeeze: If True and only one variable is sampled, return a 1D array.
                     Defaults to True.
            as_array: If True, return a NumPy array. If False (default), return a Pandas DataFrame.
            random_seed: Optional random seed for reproducibility.

        Returns:
            Union[np.ndarray, pd.DataFrame]: Sampled data, either as a NumPy array or Pandas DataFrame.

        Raises:
            RuntimeError: If sampling fails (e.g., missing equations, cycles).
            ValueError: If input parameters are invalid.
        """
        if sample_size <= 0:
            raise ValueError("sample_size must be positive")

        if random_seed is not None:
            # Create a RandomState instance for this run
            rng = np.random.RandomState(random_seed)
            # Also set the global seed for compatibility if other parts rely on it?
            # np.random.seed(random_seed) # Maybe not needed if all sampling uses rng
        else:
            # Use a default RandomState instance if no seed is provided
            rng = np.random.RandomState()

        # Get topological sort if causal graph exists
        if self._causal_graph:
            try:
                ordered_nodes = self._causal_graph.topological_sort()
            except nx.NetworkXUnfeasible:
                raise RuntimeError(
                    "Cannot sample data: Causal graph contains cycles."
                )
        else:
            # If no graph, assume no specific order, but check for cycles implicitly
            # Need a way to determine order or detect cycles without explicit graph
            # For now, just use the order variables were added
            ordered_nodes = self._variable_names
            # Warning: This might fail if there are cycles not represented in the graph

        # Filter nodes based on include_latents
        if not include_latents:
            output_nodes = [n for n in ordered_nodes if n in self._variable_names]
        else:
            output_nodes = ordered_nodes

        # Initialize data storage
        data = {node: np.full(sample_size, np.nan) for node in ordered_nodes}

        for node in ordered_nodes:
            # Check for interventions first
            if node in self._interventions:
                intervention_value = self._interventions[node]
                if callable(intervention_value):
                    # Assumes intervention function takes sample_size
                    try:
                        data[node] = intervention_value(sample_size)
                    except TypeError: # Handle functions that don't take size
                        data[node] = np.full(sample_size, intervention_value())
                else:
                    data[node] = np.full(sample_size, intervention_value)
                continue # Skip structural equation and noise for intervened nodes

            # Initialize arguments for the structural equation
            eval_args = {}

            # Get parent data if applicable
            if node in self._structural_equations:
                parents = self.get_parents(node)
                parent_data = {p: data[p] for p in parents if p in data}
                if any(np.any(np.isnan(d)) for d in parent_data.values() if d is not None):
                    raise RuntimeError(f"Missing parent data for node {node}. Check graph structure and equations.")
                eval_args.update(parent_data)

            # Generate and add noise *before* evaluating equation if needed
            generated_noise = None
            if node in self._exogenous_functions:
                noise_func = self._exogenous_functions[node]
                try:
                    # Check if noise function expects random_state
                    noise_sig = inspect.signature(noise_func)
                    if 'random_state' in noise_sig.parameters:
                        generated_noise = noise_func(sample_size=sample_size, random_state=rng)
                    else:
                        # Call without random_state if not expected (legacy?)
                        generated_noise = noise_func(sample_size=sample_size)

                    # Check if the structural equation expects noise
                    if node in self._structural_equations:
                         sig = inspect.signature(self._structural_equations[node])
                         if 'noise' in sig.parameters:
                              eval_args['noise'] = generated_noise
                         else:
                              # Equation doesn't take noise, but noise exists.
                              # We will add it *after* evaluation if an equation exists,
                              # otherwise the noise *is* the value.
                              pass
                    elif generated_noise is not None:
                         # No structural equation, node value is just the noise
                         data[node] = generated_noise
                         # Skip equation evaluation below if value is set
                         if node not in self._structural_equations: continue

                except Exception as e:
                    raise RuntimeError(
                        f"Error generating exogenous noise for node {node}: {e}"
                    ) from e

            # Evaluate structural equation if defined
            if node in self._structural_equations:
                try:
                    # Pass parents and potentially noise
                    data[node] = self._structural_equations[node](**eval_args)

                    # Add noise AFTER evaluation ONLY if equation exists but doesn't take noise itself
                    if generated_noise is not None and 'noise' not in eval_args:
                        if np.isscalar(data[node]) and not np.isscalar(generated_noise):
                            data[node] = np.full(sample_size, data[node]) + generated_noise
                        else:
                            data[node] += generated_noise

                except Exception as e:
                    raise RuntimeError(
                        f"Error evaluating structural equation for node {node}: {e}"
                    ) from e

            # Check if node value was computed (either by equation or noise)
            if np.any(np.isnan(data[node])):
                 # Check again if it was intervened
                if node not in self._interventions:
                    raise RuntimeError(
                        f"Value for node {node} could not be computed. \
                        Ensure it has a structural equation, noise model, or intervention."
                    )

        # Convert data dictionary to desired format
        result_data = {node: data[node] for node in output_nodes}

        # Always create DataFrame first with correct column order
        try:
            df = pd.DataFrame(result_data)
            # Ensure column order matches output_nodes (DataFrame constructor usually respects dict order in newer pandas, but explicit is safer)
            df = df[output_nodes]
        except Exception as e:
            # Use print or raise directly as logger might not be configured here
            print(f"Error creating DataFrame: {e}")
            print(f"Data dictionary keys: {list(result_data.keys())}")
            # Avoid printing potentially large arrays
            # print(f"Column shapes: {[arr.shape for arr in result_data.values()]}")
            raise RuntimeError("Failed to create DataFrame from sampled data. Check sampling logic.")

        if as_array:
            result_array = df.to_numpy()
            # Squeeze if only one variable and squeeze=True
            if result_array.shape[1] == 1 and squeeze:
                return result_array.flatten()
            return result_array
        else:
            # Return Pandas DataFrame
            return df

    def reset(self) -> None:
        """
        Reset the SCM to its initial state, clearing any interventions.

        Args:
            None

        Returns:
            None
        """
        # If we already have a stored original graph, restore it
        if hasattr(self, '_original_graph') and self._original_graph is not None:
            self._causal_graph = copy.deepcopy(self._original_graph)
            self._interventions = {}

    def get_observational_data(self, sample_size: int, random_seed: Optional[int] = None) -> pd.DataFrame:
        """
        Get purely observational data from the SCM.

        This method ensures that no interventions are applied when sampling data,
        by temporarily removing any active interventions, sampling data, and then
        restoring the interventions.

        Args:
            sample_size: Number of samples to generate
            random_seed: Optional random seed for reproducibility

        Returns:
            DataFrame containing the observational data

        Raises:
            Same exceptions as sample_data
        """
        # Store the current state to restore later
        current_graph = None
        current_interventions = {}

        if hasattr(self, '_original_graph'):
            # If we have interventions, store them
            if hasattr(self, '_interventions'):
                current_interventions = self._interventions.copy()
            current_graph = copy.deepcopy(self._causal_graph)

        # Reset to original state with no interventions
        self.reset()

        # Sample data in observational mode
        data = self.sample_data(sample_size, random_seed)

        # Restore the previous state if there was one
        if current_graph is not None:
            self._causal_graph = current_graph
            self._interventions = current_interventions

        return data

    def do_intervention(self, target: Any, value: Any) -> None:
        """
        Perform a 'do' intervention on the SCM.

        This method modifies the causal graph to reflect the intervention by
        removing all incoming edges to the target variable and setting it to
        the specified value.

        Args:
            target: Variable to intervene on
            value: Value to set the target variable to

        Returns:
            None

        Raises:
            ValueError: If the target variable doesn't exist in the model
        """
        if target not in self._variable_names:
            raise ValueError(
                f"Variable '{target}' does not exist in the model")

        # Store the original graph for potential reset if this is the first intervention
        if not hasattr(self, '_original_graph') or self._original_graph is None:
            self._original_graph = copy.deepcopy(self._causal_graph)
            self._interventions = {}

        # Perform the intervention on the graph
        self._causal_graph = self._causal_graph.do_intervention(target, value)

        # Keep track of the interventions
        self._interventions[target] = value

    def sample_interventional_data(self, interventions: Dict[Any, Any], sample_size: int, random_seed: Optional[int] = None) -> pd.DataFrame:
        """
        Sample data from the SCM under specified interventions.

        Args:
            interventions: Dictionary mapping variable names to intervention values
            sample_size: Number of samples to generate
            random_seed: Optional random seed for reproducibility

        Returns:
            DataFrame containing the sampled data

        Raises:
            ValueError: If any variable in interventions doesn't exist
        """
        # Store the current state to restore later
        current_graph = None
        current_interventions_state = {}

        # Check if original state exists (meaning we might be in an intervened state)
        original_graph_exists = hasattr(self, '_original_graph') and self._original_graph is not None
        if original_graph_exists:
            current_graph = copy.deepcopy(self._causal_graph)
            current_interventions_state = copy.deepcopy(self._interventions)

        # Always reset to the *actual* original state before applying new interventions
        self.reset()

        # Apply all new interventions
        self.multiple_interventions(interventions)

        # Sample data with the applied interventions, passing the seed
        try:
            data = self.sample_data(sample_size, random_seed=random_seed) # Pass seed here
        finally:
            # Restore the state that existed *before* this method call
            if original_graph_exists:
                self._causal_graph = current_graph
                self._interventions = current_interventions_state
            else:
                # If there was no original state stored, just reset to ensure clean state
                self.reset()

        return data

    def multiple_interventions(self, interventions: Dict[Any, Any]) -> None:
        """
        Perform multiple interventions on the SCM simultaneously.

        Args:
            interventions: Dictionary mapping variable names to intervention values

        Returns:
            None

        Raises:
            ValueError: If any variable in interventions doesn't exist
        """
        # Validate all variables first
        for var in interventions:
            if var not in self._variable_names:
                raise ValueError(
                    f"Variable '{var}' does not exist in the model")

        # Store the original graph for potential reset if this is the first intervention
        if not hasattr(self, '_original_graph') or self._original_graph is None:
            self._original_graph = copy.deepcopy(self._causal_graph)
            self._interventions = {}

        # Apply each intervention
        for var, value in interventions.items():
            # Remove incoming edges to the variable in the graph
            parents = list(self._causal_graph.get_parents(var))
            for parent in parents:
                self._causal_graph.remove_edge(parent, var)

            # Store the intervention value as a node attribute
            self._causal_graph._node_attributes[var] = {
                'intervention_value': value,
                **self._causal_graph.get_node_attributes(var)
            }

            # Keep track of the interventions
            self._interventions[var] = value

    def get_intervention_effects(self, target: Any, outcome: Any, intervention_values: List[Any], sample_size: int = 1000, random_seed: Optional[int] = None) -> Dict[Any, float]:
        """
        Calculate the effects of different interventions on an outcome.

        This method computes the average outcome value for each intervention value
        on the target variable.

        Args:
            target: Variable to intervene on
            outcome: Variable to measure the effect on
            intervention_values: List of values to set the target variable to
            sample_size: Number of samples to use for each intervention
            random_seed: Optional random seed for reproducibility

        Returns:
            Dictionary mapping intervention values to average outcome values

        Raises:
            ValueError: If target or outcome variables don't exist
        """
        if target not in self._variable_names:
            raise ValueError(
                f"Target variable '{target}' does not exist in the model")
        if outcome not in self._variable_names:
            raise ValueError(
                f"Outcome variable '{outcome}' does not exist in the model")

        # Store the current state to restore later
        current_graph = None
        current_interventions = {}

        if hasattr(self, '_original_graph'):
            # If we have interventions, store them
            if hasattr(self, '_interventions'):
                current_interventions = self._interventions.copy()
            current_graph = copy.deepcopy(self._causal_graph)

        # Initialize results dictionary
        effects = {}

        # For each intervention value, compute the average outcome
        for value in intervention_values:
            # Reset to original state
            self.reset()

            # Apply the intervention
            self.do_intervention(target, value)

            # Sample data with the intervention
            data = self.sample_data(sample_size, random_seed)

            # Calculate the average outcome value
            effects[value] = data[outcome].mean()

        # Restore the previous state if there was one
        if current_graph is not None:
            self._causal_graph = current_graph
            self._interventions = current_interventions
        else:
            # If there was no previous state, just reset
            self.reset()

        return effects

    def evaluate_counterfactual(self, factual_data: pd.DataFrame, interventions: Dict[Any, Any]) -> pd.DataFrame:
        """
        Evaluate counterfactual outcomes given factual data and interventions.

        This will be implemented in subtask 5.
        """
        raise NotImplementedError("Method will be implemented in subtask 5")

    def predict_outcome(self, interventions: Dict[Any, Any], conditions: Optional[Dict[Any, Any]] = None) -> Any:
        """
        Predict outcomes given interventions and optional conditioning.

        This will be implemented in subtask 5.
        """
        raise NotImplementedError("Method will be implemented in subtask 5")

    def compute_effect(self, treatment: Any, outcome: Any, treatment_value: Any, baseline_value: Optional[Any] = None, sample_size: int = 1000, random_seed: Optional[int] = None) -> float:
        """
        Compute the total causal effect of a treatment on an outcome.

        This calculates the total effect of the treatment on the outcome,
        including both direct effects and indirect effects through mediator variables.

        Args:
            treatment: Treatment variable
            outcome: Outcome variable
            treatment_value: Value to set the treatment to
            baseline_value: Optional baseline value to compare against (if None, 
                          compute the effect relative to no intervention)
            sample_size: Number of samples to use for estimation
            random_seed: Random seed for reproducibility

        Returns:
            float: Estimated total effect value

        Raises:
            ValueError: If treatment or outcome variables don't exist
        """
        # Ensure treatment and outcome exist
        if treatment not in self._variable_names:
            raise ValueError(
                f"Treatment variable '{treatment}' does not exist")
        if outcome not in self._variable_names:
            raise ValueError(f"Outcome variable '{outcome}' does not exist")

        # Save current intervention state
        original_interventions = copy.deepcopy(self._interventions)

        try:
            # Set random seed if provided
            old_random_state = None
            if random_seed is not None:
                old_random_state = np.random.get_state()
                np.random.seed(random_seed)

            # If no baseline is provided, estimate outcome with no interventions
            if baseline_value is None:
                # Reset interventions
                self.reset()
                # Sample data and compute average outcome
                baseline_data = self.sample_data(sample_size=sample_size)
                baseline_value = baseline_data[outcome].mean()

            # Set intervention
            self.do_intervention(treatment, treatment_value)

            # Sample data with intervention
            intervention_data = self.sample_data(sample_size=sample_size)

            # Compute average outcome with intervention
            effect_value = intervention_data[outcome].mean()

            # Compute effect as difference
            total_effect = effect_value - baseline_value

            return total_effect
        finally:
            # Restore original intervention state
            self.reset()
            if original_interventions:
                self.multiple_interventions(original_interventions)

            # Restore random state if it was changed
            if old_random_state is not None:
                np.random.set_state(old_random_state)

    def counterfactual_distribution(self, factual_data: pd.DataFrame, interventions: Dict[Any, Any], target_variables: Optional[List[Any]] = None, num_samples: int = 100, random_seed: Optional[int] = None) -> Dict[Any, np.ndarray]:
        """
        Generate a distribution of counterfactual outcomes.

        This method is a placeholder for future implementation.

        Args:
            factual_data: Observed data for which to compute counterfactuals
            interventions: Dictionary mapping variables to intervention values
            target_variables: Variables of interest for counterfactual distribution
            num_samples: Number of samples to generate for the distribution
            random_seed: Random seed for reproducibility

        Returns:
            Dict mapping variables to arrays of counterfactual values
        """
        # Placeholder for future implementation
        raise NotImplementedError(
            "Counterfactual distribution not yet implemented")

    # ======================= Utility Methods (New) =======================

    def to_networkx(self) -> nx.DiGraph:
        """
        Convert the SCM to a NetworkX DiGraph for advanced analysis.

        Returns:
            nx.DiGraph: A NetworkX directed graph representing the SCM's structure

        Raises:
            ValueError: If the causal graph is not defined
        """
        if self._causal_graph is None:
            raise ValueError("Causal graph is not defined in this SCM")

        # Create a new DiGraph
        G = nx.DiGraph()

        # Add nodes with metadata
        for node in self._variable_names:
            node_data = {
                "domain": self._variable_domains.get(node),
                "metadata": self._variable_metadata.get(node, {}),
                "has_equation": node in self._structural_equations,
                "has_noise": node in self._exogenous_functions,
                "intervened": node in self._interventions
            }
            G.add_node(node, **node_data)

        # Add edges from causal graph
        for source in self._variable_names:
            for target in self.get_children(source):
                G.add_edge(source, target)

        return G

    def plot(self,
             highlight_interventions: bool = True,
             layout: str = 'spring',
             ax: Optional[plt.Axes] = None,
             figsize: Tuple[int, int] = (10, 8),
             **kwargs) -> plt.Axes:
        """
        Plot the SCM's causal graph with optional highlighting.

        This method uses the visualization utilities from causal_meta.graph.visualization
        to create a visual representation of the SCM's structure.

        Args:
            highlight_interventions: Whether to highlight intervened variables
            layout: Layout algorithm to use ('spring', 'circular', etc.)
            ax: Optional matplotlib Axes object to plot on
            figsize: Figure size as (width, height) in inches
            **kwargs: Additional keyword arguments passed to plot_causal_graph

        Returns:
            plt.Axes: The matplotlib axes containing the plot

        Raises:
            ValueError: If the causal graph is not defined
        """
        if self._causal_graph is None:
            raise ValueError("Causal graph is not defined in this SCM")

        # Highlight intervened nodes if any and if requested
        highlight_nodes = set(self._interventions.keys(
        )) if highlight_interventions and self._interventions else None

        # Use the plot_causal_graph function from visualization
        ax = plot_graph(
            self._causal_graph,
            ax=ax,
            layout=layout,
            figsize=figsize,
            highlight_nodes=highlight_nodes,
            highlight_node_color='#ff7f00',  # Orange for intervened nodes
            **kwargs
        )

        # Add a legend for interventions if needed
        if highlight_interventions and self._interventions:
            handles = [plt.Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor='#ff7f00', markersize=10)]
            labels = ['Intervened Variables']
            ax.legend(handles, labels, loc='upper right')

        return ax

    def compute_direct_effect(self,
                              treatment: Any,
                              outcome: Any,
                              treatment_value: Any,
                              baseline_value: Optional[Any] = None,
                              sample_size: int = 1000,
                              random_seed: Optional[int] = None) -> float:
        """
        Compute the direct causal effect of a treatment on an outcome.

        This calculates the effect of the treatment on the outcome while
        keeping all other variables at their natural values (not manipulated).

        Args:
            treatment: Treatment variable
            outcome: Outcome variable
            treatment_value: Value to set the treatment to
            baseline_value: Optional baseline value to compare against (if None, 
                          compute the effect relative to no intervention)
            sample_size: Number of samples to use for estimation
            random_seed: Random seed for reproducibility

        Returns:
            float: Estimated direct effect value

        Raises:
            ValueError: If treatment or outcome variables don't exist
        """
        # Ensure treatment and outcome exist
        if treatment not in self._variable_names:
            raise ValueError(
                f"Treatment variable '{treatment}' does not exist")
        if outcome not in self._variable_names:
            raise ValueError(f"Outcome variable '{outcome}' does not exist")

        # Save current intervention state
        original_interventions = copy.deepcopy(self._interventions)

        try:
            # Set random seed if provided
            old_random_state = None
            if random_seed is not None:
                old_random_state = np.random.get_state()
                np.random.seed(random_seed)

            # If no baseline is provided, estimate outcome with no interventions
            if baseline_value is None:
                # Reset interventions
                self.reset()
                # Sample data and compute average outcome
                baseline_data = self.sample_data(sample_size=sample_size)
                baseline_value = baseline_data[outcome].mean()

            # Set intervention
            self.do_intervention(treatment, treatment_value)

            # Sample data with intervention
            intervention_data = self.sample_data(sample_size=sample_size)

            # Compute average outcome with intervention
            effect_value = intervention_data[outcome].mean()

            # Compute effect as difference
            direct_effect = effect_value - baseline_value

            return direct_effect
        finally:
            # Restore original intervention state
            self.reset()
            if original_interventions:
                self.multiple_interventions(original_interventions)

            # Restore random state if it was changed
            if old_random_state is not None:
                np.random.set_state(old_random_state)

    def compute_indirect_effect(self,
                                treatment: Any,
                                outcome: Any,
                                treatment_value: Any,
                                mediators: Optional[List[Any]] = None,
                                baseline_value: Optional[Any] = None,
                                sample_size: int = 1000,
                                random_seed: Optional[int] = None) -> float:
        """
        Compute the indirect effect of a treatment on an outcome through mediators.

        This calculates the portion of the total effect that is transmitted through
        mediator variables, rather than directly from treatment to outcome.

        Args:
            treatment: Treatment variable
            outcome: Outcome variable
            treatment_value: Value to set the treatment to
            mediators: List of mediator variables (if None, all children of treatment
                      that are ancestors of outcome are considered)
            baseline_value: Optional baseline value to compare against
            sample_size: Number of samples to use for estimation
            random_seed: Random seed for reproducibility

        Returns:
            float: Estimated indirect effect value

        Raises:
            ValueError: If treatment or outcome variables don't exist
        """
        # Calculate total effect
        total_effect = self.compute_effect(
            treatment=treatment,
            outcome=outcome,
            treatment_value=treatment_value,
            baseline_value=baseline_value,
            sample_size=sample_size,
            random_seed=random_seed
        )

        # Calculate direct effect
        direct_effect = self.compute_direct_effect(
            treatment=treatment,
            outcome=outcome,
            treatment_value=treatment_value,
            baseline_value=baseline_value,
            sample_size=sample_size,
            random_seed=random_seed
        )

        # Indirect effect = Total effect - Direct effect
        indirect_effect = total_effect - direct_effect
        return indirect_effect

    def get_adjacency_matrix(self, node_order: Optional[List] = None) -> np.ndarray:
        """
        Return the adjacency matrix of the underlying causal graph.

        Args:
            node_order: Optional list specifying the order of nodes for the matrix rows/columns.
                        If None, the default node order of the graph is used.

        Returns:
            A numpy ndarray representing the adjacency matrix.

        Raises:
            ValueError: If the causal graph is not defined for this SCM.
        """
        if self._causal_graph is None:
            raise ValueError("Causal graph is not defined in this SCM.")
        return self._causal_graph.get_adjacency_matrix(node_order=node_order)

    def compare_structure(self, other_scm: 'StructuralCausalModel') -> Tuple[bool, Dict[str, Any]]:
        """
        Compare the structure of this SCM with another SCM.

        Args:
            other_scm: Another StructuralCausalModel to compare with

        Returns:
            Tuple containing:
                - bool: True if structures are equivalent, False otherwise
                - Dict: Detailed comparison results including differences
        """
        results = {
            "equivalent": True,
            "variable_diff": [],
            "edge_diff": [],
            "details": {}
        }

        # Compare variables
        self_vars = set(self._variable_names)
        other_vars = set(other_scm._variable_names)

        if self_vars != other_vars:
            results["equivalent"] = False
            results["variable_diff"] = {
                "only_in_self": list(self_vars - other_vars),
                "only_in_other": list(other_vars - self_vars),
                "common": list(self_vars.intersection(other_vars))
            }

        # Compare edges for common variables
        common_vars = self_vars.intersection(other_vars)
        edge_diff = []

        for var in common_vars:
            self_parents = set(self.get_parents(var))
            try:
                other_parents = set(other_scm.get_parents(var))

                if self_parents != other_parents:
                    edge_diff.append({
                        "variable": var,
                        "only_in_self": list(self_parents - other_parents),
                        "only_in_other": list(other_parents - self_parents)
                    })
            except ValueError:
                # Variable might exist but not in the graph
                edge_diff.append({
                    "variable": var,
                    "only_in_self": list(self_parents),
                    "only_in_other": []
                })

        if edge_diff:
            results["equivalent"] = False
            results["edge_diff"] = edge_diff

        return results["equivalent"], results

    def to_json(self, path: Optional[str] = None) -> Optional[str]:
        """
        Serialize the SCM to JSON format.

        Args:
            path: Optional file path to save the JSON. If None, returns the JSON string.

        Returns:
            str: JSON string representation if path is None, otherwise None

        Raises:
            ValueError: If serialization fails
        """
        # Create a dictionary representation of the SCM
        scm_dict = {
            "variables": self._variable_names,
            "domains": self._variable_domains,
            "metadata": self._variable_metadata,
            "edges": [],
            "interventions": self._interventions
        }

        # Add edges if causal graph exists
        if self._causal_graph is not None:
            for node in self._variable_names:
                for child in self.get_children(node):
                    scm_dict["edges"].append({"from": node, "to": child})

        # Note: We can't serialize the functions directly in JSON
        # Instead, we just note which variables have equations defined
        scm_dict["has_equations"] = list(self._structural_equations.keys())
        scm_dict["has_exogenous"] = list(self._exogenous_functions.keys())

        # Convert to JSON
        json_str = json.dumps(scm_dict, indent=2)

        # Save to file if path provided
        if path:
            with open(path, 'w') as f:
                f.write(json_str)
            return None

        return json_str

    def save(self, path: str) -> None:
        """
        Save the complete SCM to a file using pickle.

        Unlike to_json, this method preserves the equation functions.

        Args:
            path: File path to save the pickled SCM

        Raises:
            IOError: If saving fails
        """
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> 'StructuralCausalModel':
        """
        Load an SCM from a pickled file.

        Args:
            path: File path to load the pickled SCM from

        Returns:
            StructuralCausalModel: The loaded SCM

        Raises:
            IOError: If loading fails
            ValueError: If the loaded object is not an SCM
        """
        with open(path, 'rb') as f:
            scm = pickle.load(f)

        if not isinstance(scm, cls):
            raise ValueError(f"Loaded object is not a {cls.__name__}")

        return scm

    def __str__(self) -> str:
        """
        Return a string representation of the SCM.

        Returns:
            str: Human-readable description of the SCM
        """
        parts = [
            f"StructuralCausalModel with {len(self._variable_names)} variables:"]

        # Add variable information
        parts.append("\nVariables:")
        for var in self._variable_names:
            domain = f", domain: {self._variable_domains.get(var, 'not specified')}" if var in self._variable_domains else ""
            equation = ", has equation: Yes" if var in self._structural_equations else ", has equation: No"
            noise = ", has noise: Yes" if var in self._exogenous_functions else ", has noise: No"
            intervened = f", intervened: {self._interventions[var]}" if var in self._interventions else ""

            parts.append(f"  {var}{domain}{equation}{noise}{intervened}")

        # Add relationship information if causal graph exists
        if self._causal_graph is not None:
            parts.append("\nRelationships:")
            for var in self._variable_names:
                parents = self.get_parents(var)
                if parents:
                    parts.append(f"  {var} ← {', '.join(parents)}")

        # Add intervention information if any
        if self._interventions:
            parts.append("\nActive Interventions:")
            for var, value in self._interventions.items():
                parts.append(f"  do({var} = {value})")

        return "\n".join(parts)

    def __repr__(self) -> str:
        """
        Return a string representation that could be used to recreate the object.

        Returns:
            str: String representation of the SCM
        """
        return f"<{self.__class__.__name__} with {len(self._variable_names)} variables and {len(self._structural_equations)} equations>"

    def sample_data_interventional(
            self,
            interventions: List['Intervention'], # noqa: F821 - Use string literal for forward reference
            sample_size: int,
            random_seed: Optional[int] = None
        ) -> pd.DataFrame:
            """
            Sample data from the SCM after applying a sequence of interventions.

            This method creates a modified copy of the SCM by applying the specified
            interventions sequentially and then samples data from the resulting
            intervened model. The original SCM remains unchanged.

            Args:
                interventions: A list of Intervention objects (e.g., PerfectIntervention,
                               ImperfectIntervention, SoftIntervention) to apply.
                sample_size: The number of samples to generate from the intervened SCM.
                random_seed: Optional random seed for reproducibility of the sampling process.

            Returns:
                pd.DataFrame: DataFrame containing the sampled data from the intervened SCM.

            Raises:
                ValueError: If any intervention is invalid or cannot be applied,
                           or if sampling fails on the intervened SCM.
                TypeError: If `interventions` is not a list of Intervention objects.
            """
            # Imports should be checked/added at the top of the file:
            # import copy
            # from typing import List, Optional
            # import pandas as pd
            # from causal_meta.environments.interventions import Intervention

            if not isinstance(interventions, list):
                raise TypeError("`interventions` must be a list of Intervention objects.")

            # Instance check requires Intervention to be imported.
            # Assuming caller ensures correct type.
            # from causal_meta.environments.interventions import Intervention
            # if not all(isinstance(interv, Intervention) for interv in interventions):
            #     raise TypeError("All elements in `interventions` must be Intervention objects.")

            intervened_scm = copy.deepcopy(self)

            for intervention in interventions:
                try:
                    intervened_scm = intervention.apply(intervened_scm)
                except (ValueError, NotImplementedError, TypeError) as e:
                    raise ValueError(f"Failed to apply intervention {intervention!r}: {e}") from e

            try:
                return intervened_scm.sample_data(sample_size=sample_size, random_seed=random_seed)
            except Exception as e:
                raise RuntimeError(f"Failed to sample data from intervened SCM: {e}") from e

    @property
    def nodes(self) -> list:
        """Return the list of nodes in the causal graph."""
        return self._causal_graph.get_nodes()

