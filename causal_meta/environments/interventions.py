from abc import ABC, abstractmethod
from typing import Any, Dict, TYPE_CHECKING
import copy
import inspect

if TYPE_CHECKING:
    from causal_meta.environments.scm import StructuralCausalModel


class Intervention(ABC):
    """Abstract base class for all intervention types."""

    def __init__(self, target_node: str, **kwargs: Any) -> None:
        """Initialize the intervention.

        Args:
            target_node: The name of the node to intervene on.
            **kwargs: Additional keyword arguments specific to the intervention type.
        """
        if not isinstance(target_node, str) or not target_node:
            raise ValueError("target_node must be a non-empty string.")
        self.target_node = target_node
        self._validate_kwargs(**kwargs)

    def _validate_kwargs(self, **kwargs: Any) -> None:
        """Validate intervention-specific keyword arguments.

        Subclasses should override this method to validate their specific parameters.

        Args:
            **kwargs: Keyword arguments passed during initialization.

        Raises:
            ValueError: If any keyword argument is invalid.
        """
        # Base implementation does nothing; subclasses add specific validation.
        pass

    @abstractmethod
    def apply(self, scm: 'StructuralCausalModel') -> 'StructuralCausalModel':
        """Apply the intervention to a given Structural Causal Model (SCM).

        This method should modify the SCM according to the intervention logic.
        It should typically return a modified *copy* of the SCM to avoid
        altering the original model in place, unless explicitly designed to do so.

        Args:
            scm: The StructuralCausalModel instance to apply the intervention to.

        Returns:
            The modified StructuralCausalModel.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
            ValueError: If the intervention cannot be applied to the given SCM
                        (e.g., target node does not exist).
        """
        raise NotImplementedError

    @abstractmethod
    def __repr__(self) -> str:
        """Return a string representation of the intervention."""
        raise NotImplementedError

    # Potentially add other common methods like reversing an intervention if needed.


class PerfectIntervention(Intervention):
    """Represents a perfect intervention (do-operator).

    This intervention sets the target node to a fixed value, removing its
    dependence on its causal parents.
    """

    def __init__(self, target_node: str, value: Any) -> None:
        """Initialize the perfect intervention.

        Args:
            target_node: The name of the node to intervene on.
            value: The fixed value to assign to the target node.
        """
        super().__init__(target_node=target_node, value=value)
        self.value = value

    def _validate_kwargs(self, **kwargs: Any) -> None:
        """Validate that 'value' is provided."""
        if 'value' not in kwargs:
            raise ValueError(
                "PerfectIntervention requires a 'value' keyword argument.")
        # Further type validation for self.value could be added here if needed.

    def apply(self, scm: 'StructuralCausalModel') -> 'StructuralCausalModel':
        """Apply the perfect intervention to the SCM.

        Creates a copy of the SCM, removes the target node's parents,
        and replaces its structural equation with a constant function.

        Args:
            scm: The StructuralCausalModel instance.

        Returns:
            A new SCM instance with the intervention applied.

        Raises:
            ValueError: If the target node is not found in the SCM.
        """
        graph = scm.get_causal_graph()
        if graph is None:
            raise ValueError("SCM does not have a causal graph defined.")

        if self.target_node not in graph.get_nodes():  # Use getter
            raise ValueError(
                f"Target node '{self.target_node}' not found in SCM graph.")

        # Create a deep copy to avoid modifying the original SCM
        scm_copy = copy.deepcopy(scm)
        graph_copy = scm_copy.get_causal_graph()  # Get graph from copy

        # Remove parents of the target node in the copied graph
        parents = list(graph_copy.get_parents(self.target_node))
        for parent in parents:
            graph_copy.remove_edge(parent, self.target_node)

        # Replace the structural equation with a constant function
        def constant_mechanism(**parent_values: Any) -> Any:
            return self.value

        scm_copy._structural_equations[self.target_node] = constant_mechanism

        # Update exogenous variable mapping if necessary (assuming noise is handled elsewhere)
        # scm_copy.exogenous_variables[self.target_node] = None # Or handle appropriately

        return scm_copy

    def __repr__(self) -> str:
        """Return a string representation of the perfect intervention."""
        return f"PerfectIntervention(target_node='{self.target_node}', value={self.value!r})"


class ImperfectIntervention(Intervention):
    """Represents an imperfect intervention.

    Modifies the target node's value while potentially retaining some
    influence from its original structural equation and parents.
    """

    def __init__(self, target_node: str, value: Any, strength: float = 1.0, combination_method: str = 'additive') -> None:
        """Initialize the imperfect intervention.

        Args:
            target_node: The name of the node to intervene on.
            value: The intervention value to incorporate.
            strength: The strength of the intervention (0 to 1). Default 1.0.
                      Interpretation depends on combination_method.
                      For 'additive': coefficient for intervention value.
                      For 'multiplicative': coefficient for intervention value.
                      For 'weighted_average': weight for intervention value (1-strength for original).
            combination_method: How to combine the original mechanism and intervention value.
                                Options: 'additive', 'multiplicative', 'weighted_average'.
                                Default 'additive'.
        """
        super().__init__(target_node=target_node, value=value,
                         strength=strength, combination_method=combination_method)
        self.value = value
        self.strength = strength
        self.combination_method = combination_method

    def _validate_kwargs(self, **kwargs: Any) -> None:
        """Validate intervention parameters."""
        if 'value' not in kwargs:
            raise ValueError(
                "ImperfectIntervention requires a 'value' keyword argument.")
        strength = kwargs.get('strength', 1.0)
        if not isinstance(strength, (int, float)) or not (0.0 <= strength <= 1.0):
            raise ValueError("strength must be a float between 0.0 and 1.0.")
        combination_method = kwargs.get('combination_method', 'additive')
        if combination_method not in ['additive', 'multiplicative', 'weighted_average']:
            raise ValueError(
                "combination_method must be one of 'additive', 'multiplicative', 'weighted_average'.")

    def apply(self, scm: 'StructuralCausalModel') -> 'StructuralCausalModel':
        """Apply the imperfect intervention to the SCM.

        Creates a copy of the SCM and replaces the target node's structural
        equation with a new one combining the original mechanism and the
        intervention value.

        Args:
            scm: The StructuralCausalModel instance.

        Returns:
            A new SCM instance with the intervention applied.

        Raises:
            ValueError: If the target node or its original mechanism is not found.
        """
        graph = scm.get_causal_graph()
        if graph is None:
            raise ValueError("SCM does not have a causal graph defined.")

        if self.target_node not in graph.get_nodes():  # Use getter
            raise ValueError(
                f"Target node '{self.target_node}' not found in SCM graph.")
        if self.target_node not in scm._structural_equations:  # Check internal dict ok here
            raise ValueError(
                f"Original structural equation for '{self.target_node}' not found in SCM.")

        scm_copy = copy.deepcopy(scm)
        # Use the internal _structural_equations attribute
        original_mechanism = scm_copy._structural_equations[self.target_node]
        original_sig = inspect.signature(original_mechanism)
        expects_noise = 'noise' in original_sig.parameters

        # Define the new mechanism based on the combination method
        if self.combination_method == 'additive':
            if expects_noise:
                def imperfect_mechanism(noise: Any = None, **kwargs: Any) -> Any:
                    parent_values = kwargs # Noise is now separate
                    noise_value = noise # Get noise directly from argument

                    # Prepare args for original mechanism
                    original_args = parent_values.copy()
                    if noise_value is not None:
                        original_args['noise'] = noise_value
                    # No need to check sig again, we know it expects noise

                    try:
                        original_value = original_mechanism(**original_args)
                    except TypeError as e:
                        # Provide more detailed error info
                        expected_params = list(original_sig.parameters.keys())
                        provided_args = list(original_args.keys())
                        raise TypeError(
                            f"Error calling original additive mechanism for {self.target_node}. "
                            f"Expected params: {expected_params}. Provided args: {provided_args}. "
                            f"Original error: {e}"
                        ) from e

                    # Ensure values are numeric for combination
                    try:
                        original_value_num = float(original_value)
                        intervention_value_num = float(self.value)
                    except (ValueError, TypeError) as e:
                        raise TypeError(f"Cannot apply additive intervention to non-numeric values: original={original_value}, intervention={self.value}") from e

                    return (1 - self.strength) * original_value_num + self.strength * intervention_value_num
            else:
                # Original mechanism does not expect noise
                def imperfect_mechanism(**kwargs: Any) -> Any:
                    parent_values = kwargs
                    # No noise handling needed for original_mechanism call

                    try:
                        original_value = original_mechanism(**parent_values)
                    except TypeError as e:
                        expected_params = list(original_sig.parameters.keys())
                        provided_args = list(parent_values.keys())
                        raise TypeError(
                            f"Error calling original additive mechanism (no noise expected) for {self.target_node}. "
                            f"Expected params: {expected_params}. Provided args: {provided_args}. "
                            f"Original error: {e}"
                        ) from e

                    # Ensure values are numeric for combination
                    try:
                        original_value_num = float(original_value)
                        intervention_value_num = float(self.value)
                    except (ValueError, TypeError) as e:
                        raise TypeError(f"Cannot apply additive intervention to non-numeric values: original={original_value}, intervention={self.value}") from e

                    return (1 - self.strength) * original_value_num + self.strength * intervention_value_num

        elif self.combination_method == 'multiplicative':
            if expects_noise:
                def imperfect_mechanism(noise: Any = None, **kwargs: Any) -> Any:
                    parent_values = kwargs
                    noise_value = noise

                    original_args = parent_values.copy()
                    if noise_value is not None:
                        original_args['noise'] = noise_value

                    try:
                        original_value = original_mechanism(**original_args)
                    except TypeError as e:
                        expected_params = list(original_sig.parameters.keys())
                        provided_args = list(original_args.keys())
                        raise TypeError(
                            f"Error calling original multiplicative mechanism for {self.target_node}. "
                            f"Expected params: {expected_params}. Provided args: {provided_args}. "
                            f"Original error: {e}"
                        ) from e

                    # Ensure values are numeric for combination
                    try:
                        original_value_num = float(original_value)
                        intervention_value_num = float(self.value)
                    except (ValueError, TypeError) as e:
                        raise TypeError(f"Cannot apply multiplicative intervention to non-numeric values: original={original_value}, intervention={self.value}") from e

                    return original_value_num**(1 - self.strength) * intervention_value_num**self.strength
            else:
                # Original mechanism does not expect noise
                def imperfect_mechanism(**kwargs: Any) -> Any:
                    parent_values = kwargs
                    try:
                        original_value = original_mechanism(**parent_values)
                    except TypeError as e:
                        expected_params = list(original_sig.parameters.keys())
                        provided_args = list(parent_values.keys())
                        raise TypeError(
                            f"Error calling original multiplicative mechanism (no noise expected) for {self.target_node}. "
                            f"Expected params: {expected_params}. Provided args: {provided_args}. "
                            f"Original error: {e}"
                        ) from e

                    # Ensure values are numeric for combination
                    try:
                        original_value_num = float(original_value)
                        intervention_value_num = float(self.value)
                    except (ValueError, TypeError) as e:
                        raise TypeError(f"Cannot apply multiplicative intervention to non-numeric values: original={original_value}, intervention={self.value}") from e

                    return original_value_num**(1 - self.strength) * intervention_value_num**self.strength

        elif self.combination_method == 'weighted_average':
            # Weighted average is mathematically the same as additive for strength s
            if expects_noise:
                 def imperfect_mechanism(noise: Any = None, **kwargs: Any) -> Any:
                    parent_values = kwargs
                    noise_value = noise

                    original_args = parent_values.copy()
                    if noise_value is not None:
                        original_args['noise'] = noise_value

                    try:
                        original_value = original_mechanism(**original_args)
                    except TypeError as e:
                        expected_params = list(original_sig.parameters.keys())
                        provided_args = list(original_args.keys())
                        raise TypeError(
                            f"Error calling original weighted_average mechanism for {self.target_node}. "
                            f"Expected params: {expected_params}. Provided args: {provided_args}. "
                            f"Original error: {e}"
                        ) from e

                    # Ensure values are numeric for combination
                    try:
                        original_value_num = float(original_value)
                        intervention_value_num = float(self.value)
                    except (ValueError, TypeError) as e:
                        raise TypeError(f"Cannot apply weighted_average intervention to non-numeric values: original={original_value}, intervention={self.value}") from e

                    return (1 - self.strength) * original_value_num + self.strength * intervention_value_num
            else:
                # Original mechanism does not expect noise
                def imperfect_mechanism(**kwargs: Any) -> Any:
                    parent_values = kwargs
                    try:
                        original_value = original_mechanism(**parent_values)
                    except TypeError as e:
                        expected_params = list(original_sig.parameters.keys())
                        provided_args = list(parent_values.keys())
                        raise TypeError(
                            f"Error calling original weighted_average mechanism (no noise expected) for {self.target_node}. "
                            f"Expected params: {expected_params}. Provided args: {provided_args}. "
                            f"Original error: {e}"
                        ) from e

                    # Ensure values are numeric for combination
                    try:
                        original_value_num = float(original_value)
                        intervention_value_num = float(self.value)
                    except (ValueError, TypeError) as e:
                        raise TypeError(f"Cannot apply weighted_average intervention to non-numeric values: original={original_value}, intervention={self.value}") from e

                    return (1 - self.strength) * original_value_num + self.strength * intervention_value_num
        else:
            # Should be caught by validation, but added for safety
            raise ValueError(f"Unsupported combination method: {self.combination_method}")

        # Set the new mechanism in the copied SCM
        scm_copy._structural_equations[self.target_node] = imperfect_mechanism
        # Parent edges are NOT removed for imperfect interventions

        return scm_copy

    def __repr__(self) -> str:
        """Return a string representation of the imperfect intervention."""
        return (f"ImperfectIntervention(target_node='{self.target_node}', "
                f"value={self.value!r}, strength={self.strength}, "
                f"combination_method='{self.combination_method}')")


class SoftIntervention(Intervention):
    """Represents a soft intervention.

    Modifies the mechanism (structural equation) of the target node using a
    provided function, potentially composing it with the original mechanism.
    """

    def __init__(self, target_node: str, intervention_function: callable, compose: bool = False, **func_kwargs: Any) -> None:
        """Initialize the soft intervention.

        Args:
            target_node: The name of the node to intervene on.
            intervention_function: A callable function that defines the new
                                   mechanism or a modification to the old one.
                                   Its signature should match the expected input
                                   for the node's structural equation (i.e.,
                                   accept parent values as keyword arguments).
            compose: If True, the intervention_function will be composed with
                     the original mechanism (applied after). If False (default),
                     it replaces the original mechanism.
            **func_kwargs: Additional keyword arguments to be passed to the
                           intervention_function during its execution.
        """
        super().__init__(target_node=target_node,
                         intervention_function=intervention_function,
                         compose=compose, **func_kwargs)
        if not callable(intervention_function):
            raise ValueError("intervention_function must be a callable.")
        self.intervention_function = intervention_function
        self.compose = compose
        self.func_kwargs = func_kwargs  # Store additional kwargs

    def _validate_kwargs(self, **kwargs: Any) -> None:
        """Validate intervention parameters."""
        if 'intervention_function' not in kwargs or not callable(kwargs['intervention_function']):
            raise ValueError(
                "SoftIntervention requires a callable 'intervention_function' keyword argument.")
        if 'compose' in kwargs and not isinstance(kwargs['compose'], bool):
            raise ValueError("'compose' must be a boolean value.")
        # No specific validation needed for func_kwargs here, handled by function call later

    def apply(self, scm: 'StructuralCausalModel') -> 'StructuralCausalModel':
        """Apply the soft intervention to the SCM.

        Creates a copy of the SCM and replaces or composes the target node's
        structural equation with the provided intervention function.

        Args:
            scm: The StructuralCausalModel instance.

        Returns:
            A new SCM instance with the intervention applied.

        Raises:
            ValueError: If the target node or its original mechanism (if compose=True)
                        is not found.
        """
        graph = scm.get_causal_graph()
        if graph is None:
            raise ValueError("SCM does not have a causal graph defined.")

        if self.target_node not in graph.get_nodes():  # Use getter
            raise ValueError(
                f"Target node '{self.target_node}' not found in SCM graph.")

        scm_copy = copy.deepcopy(scm)

        if self.compose:
            # Use the internal _structural_equations attribute
            if self.target_node not in scm_copy._structural_equations:
                raise ValueError(
                    f"Original structural equation for '{self.target_node}' not found in SCM for composition.")
            original_mechanism = scm_copy._structural_equations[self.target_node]

            def composed_mechanism(**parent_values: Any) -> Any:
                # Compute original value first
                original_value = original_mechanism(**parent_values)
                # Apply intervention function to the original value
                # Note: We might need a way to pass parent_values too if needed by intervention_function
                # For now, assume intervention_function operates on the intermediate result
                # Or maybe it should take original_value AND parent_values? Let's assume it takes parents for flexibility.
                # **self.func_kwargs allows passing fixed parameters during init
                return self.intervention_function(original_output=original_value, **parent_values, **self.func_kwargs)

            # Use the internal _structural_equations attribute
            scm_copy._structural_equations[self.target_node] = composed_mechanism
        else:
            # Replace mechanism directly
            def new_mechanism(**parent_values: Any) -> Any:
                return self.intervention_function(**parent_values, **self.func_kwargs)

            # Use the internal _structural_equations attribute
            scm_copy._structural_equations[self.target_node] = new_mechanism

        # Parent edges are NOT removed for soft interventions

        return scm_copy

    def __repr__(self) -> str:
        """Return a string representation of the soft intervention."""
        func_name = getattr(self.intervention_function,
                            '__name__', repr(self.intervention_function))
        return (f"SoftIntervention(target_node='{self.target_node}', "
                f"intervention_function={func_name}, compose={self.compose}, "
                f"func_kwargs={self.func_kwargs!r})")
