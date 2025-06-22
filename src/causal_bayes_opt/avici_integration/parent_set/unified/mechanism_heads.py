"""
Optional mechanism prediction heads for the unified model.

These heads can be enabled via feature flags to predict mechanism information
in addition to parent set structure.
"""

from typing import Dict, List, Any
import jax
import jax.numpy as jnp
import haiku as hk
from dataclasses import dataclass


@dataclass(frozen=True)
class MechanismPrediction:
    """Structured mechanism prediction for a parent set."""
    parent_set_idx: int
    mechanism_type: str
    mechanism_confidence: float
    parameters: Dict[str, Any]
    uncertainty: float


class MechanismPredictionHeads(hk.Module):
    """
    Optional mechanism prediction heads that can be added to the unified model.
    
    These heads predict HOW parent variables influence the target,
    not just WHICH variables are parents.
    """
    
    def __init__(self, config, name="MechanismPredictionHeads"):
        super().__init__(name=name)
        self.config = config
        self.w_init = hk.initializers.VarianceScaling(2.0, "fan_in", "uniform")
    
    def __call__(self, 
                 pooled_features: jnp.ndarray,  # [hidden_dim]
                 parent_set_logits: jnp.ndarray,  # [k]
                 parent_sets: List[frozenset]  # List of k parent sets
                 ) -> Dict[str, jnp.ndarray]:
        """
        Predict mechanism information for each parent set.
        
        Args:
            pooled_features: Global dataset features
            parent_set_logits: Logits for each parent set
            parent_sets: List of parent sets being considered
            
        Returns:
            Dictionary with mechanism predictions
        """
        if not self.config.predict_mechanisms:
            return {}
        
        k = len(parent_sets)
        n_mechanism_types = len(self.config.mechanism_types)
        
        # Mechanism type classification for each parent set
        mechanism_type_logits = self._predict_mechanism_types(
            pooled_features, k, n_mechanism_types
        )
        
        # Mechanism parameter regression
        mechanism_parameters = self._predict_mechanism_parameters(
            pooled_features, k, n_mechanism_types
        )
        
        # Uncertainty quantification (optional)
        uncertainties = self._predict_uncertainties(
            pooled_features, k, n_mechanism_types
        ) if self.config.enable_uncertainty_quantification else None
        
        output = {
            "mechanism_type_logits": mechanism_type_logits,  # [k, n_types]
            "mechanism_parameters": mechanism_parameters,    # [k, n_types, param_dim]
        }
        
        if uncertainties is not None:
            output["mechanism_uncertainties"] = uncertainties  # [k, n_types]
            
        return output
    
    def _predict_mechanism_types(self, 
                                features: jnp.ndarray, 
                                k: int, 
                                n_types: int) -> jnp.ndarray:
        """Predict mechanism type for each parent set."""
        mechanism_classifier = hk.Sequential([
            hk.Linear(self.config.dim, w_init=self.w_init, name="mech_type_1"),
            jax.nn.relu,
            hk.Linear(self.config.dim // 2, w_init=self.w_init, name="mech_type_2"),
            jax.nn.relu,
            hk.Linear(k * n_types, w_init=self.w_init, name="mech_type_out")
        ], name="mechanism_type_classifier")
        
        logits = mechanism_classifier(features)
        return jnp.reshape(logits, (k, n_types))
    
    def _predict_mechanism_parameters(self, 
                                    features: jnp.ndarray, 
                                    k: int, 
                                    n_types: int) -> jnp.ndarray:
        """Predict mechanism parameters for each parent set and mechanism type."""
        parameter_regressor = hk.Sequential([
            hk.Linear(self.config.dim, w_init=self.w_init, name="mech_param_1"),
            jax.nn.relu,
            hk.Linear(self.config.dim // 2, w_init=self.w_init, name="mech_param_2"),
            jax.nn.relu,
            hk.Linear(k * n_types * self.config.mechanism_param_dim, 
                     w_init=self.w_init, name="mech_param_out")
        ], name="mechanism_parameter_regressor")
        
        params = parameter_regressor(features)
        return jnp.reshape(params, (k, n_types, self.config.mechanism_param_dim))
    
    def _predict_uncertainties(self, 
                              features: jnp.ndarray, 
                              k: int, 
                              n_types: int) -> jnp.ndarray:
        """Predict uncertainty estimates for mechanism predictions."""
        uncertainty_estimator = hk.Sequential([
            hk.Linear(self.config.dim // 2, w_init=self.w_init, name="unc_1"),
            jax.nn.relu,
            hk.Linear(k * n_types, w_init=self.w_init, name="unc_out"),
            jax.nn.softplus  # Ensure positive uncertainties
        ], name="uncertainty_estimator")
        
        uncertainties = uncertainty_estimator(features)
        return jnp.reshape(uncertainties, (k, n_types))


def interpret_mechanism_predictions(
    mechanism_outputs: Dict[str, jnp.ndarray],
    parent_sets: List[frozenset],
    config
) -> List[MechanismPrediction]:
    """
    Convert raw mechanism outputs to structured predictions.
    
    Args:
        mechanism_outputs: Raw outputs from mechanism heads
        parent_sets: List of parent sets
        config: Configuration with mechanism types
        
    Returns:
        List of structured mechanism predictions
    """
    if not mechanism_outputs:
        return []
    
    mechanism_type_logits = mechanism_outputs["mechanism_type_logits"]  # [k, n_types]
    mechanism_parameters = mechanism_outputs["mechanism_parameters"]    # [k, n_types, param_dim]
    uncertainties = mechanism_outputs.get("mechanism_uncertainties")   # [k, n_types] or None
    
    predictions = []
    
    for i, parent_set in enumerate(parent_sets):
        # Get most likely mechanism type
        type_probs = jax.nn.softmax(mechanism_type_logits[i])
        best_type_idx = jnp.argmax(type_probs)
        best_type = config.mechanism_types[int(best_type_idx)]
        confidence = float(type_probs[best_type_idx])
        
        # Extract parameters for best mechanism type
        raw_params = mechanism_parameters[i, best_type_idx]
        parsed_params = _parse_mechanism_parameters(raw_params, best_type)
        
        # Get uncertainty if available
        uncertainty = float(uncertainties[i, best_type_idx]) if uncertainties is not None else 0.0
        
        prediction = MechanismPrediction(
            parent_set_idx=i,
            mechanism_type=best_type,
            mechanism_confidence=confidence,
            parameters=parsed_params,
            uncertainty=uncertainty
        )
        predictions.append(prediction)
    
    return predictions


def _parse_mechanism_parameters(raw_params: jnp.ndarray, mechanism_type: str) -> Dict[str, Any]:
    """
    Parse raw parameter vector into mechanism-specific parameters.
    
    Args:
        raw_params: Raw parameter vector [param_dim]
        mechanism_type: Type of mechanism
        
    Returns:
        Dictionary of parsed parameters
    """
    params = {}
    
    if mechanism_type == "linear":
        # For linear mechanisms: coefficients and intercept
        params["coefficients"] = raw_params[:-1].tolist()  # All but last element
        params["intercept"] = float(raw_params[-1])        # Last element
        
    elif mechanism_type == "polynomial":
        # For polynomial mechanisms: coefficients for different powers
        mid = len(raw_params) // 2
        params["linear_coeffs"] = raw_params[:mid].tolist()
        params["quadratic_coeffs"] = raw_params[mid:].tolist()
        
    elif mechanism_type == "gaussian":
        # For Gaussian mechanisms: mean, variance parameters
        params["mean_params"] = raw_params[:len(raw_params)//2].tolist()
        params["var_params"] = jax.nn.softplus(raw_params[len(raw_params)//2:]).tolist()
        
    elif mechanism_type == "neural":
        # For neural mechanisms: hidden layer weights (simplified)
        params["hidden_weights"] = raw_params.tolist()
        
    else:
        # Default: just store raw parameters
        params["raw_parameters"] = raw_params.tolist()
    
    return params


def compute_mechanism_impact_scores(
    predictions: List[MechanismPrediction],
    parent_sets: List[frozenset]
) -> jnp.ndarray:
    """
    Compute impact scores for each parent set based on mechanism predictions.
    
    This helps the acquisition policy prioritize high-impact interventions.
    
    Args:
        predictions: List of mechanism predictions
        parent_sets: List of parent sets
        
    Returns:
        Impact scores [k] for each parent set
    """
    k = len(parent_sets)
    scores = jnp.zeros(k)
    
    for i, (pred, parent_set) in enumerate(zip(predictions, parent_sets)):
        # Base score from mechanism confidence
        base_score = pred.mechanism_confidence
        
        # Adjust based on mechanism type (some types have higher impact)
        type_multiplier = {
            "linear": 1.0,
            "polynomial": 1.2,      # Non-linear effects are more informative
            "gaussian": 1.1,
            "neural": 1.3           # Complex mechanisms are most informative
        }.get(pred.mechanism_type, 1.0)
        
        # Adjust based on parent set size (more parents = more complex interactions)
        size_multiplier = 1.0 + 0.1 * len(parent_set)
        
        # Adjust based on uncertainty (higher uncertainty = more information gain potential)
        uncertainty_bonus = 1.0 + pred.uncertainty
        
        final_score = base_score * type_multiplier * size_multiplier * uncertainty_bonus
        scores = scores.at[i].set(final_score)
    
    return scores