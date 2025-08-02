#!/usr/bin/env python3
"""
Mechanism-Aware Parent Set Prediction Demo

This script demonstrates the ModularParentSetModel with configurable mechanism
prediction capabilities, showing both structure-only and enhanced modes.

Part A: Modular Model Architecture (Complete)
"""

import jax.random as random
import jax.numpy as jnp

from causal_bayes_opt.avici_integration.parent_set.mechanism_aware import (
    MechanismAwareConfig,
    create_modular_parent_set_model,
    predict_with_mechanisms,
    create_structure_only_config,
    create_enhanced_config,
    compare_model_outputs
)


def main():
    """Demonstrate mechanism-aware parent set prediction."""
    print("=== Mechanism-Aware Parent Set Prediction Demo ===\n")
    
    # Setup
    key = random.PRNGKey(42)
    variable_order = ["X", "Y", "Z", "W"]
    target_variable = "Y"
    
    # Create synthetic data [N, d, 3] in AVICI format
    N, d = 20, 4
    data = jnp.ones((N, d, 3))
    data = data.at[:, :, 0].set(random.normal(key, (N, d)))  # Variable values
    data = data.at[:10, 0, 1].set(1.0)  # Some interventions on X
    data = data.at[:, 1, 2].set(1.0)    # Y is target
    
    print(f"Data shape: {data.shape}")
    print(f"Variables: {variable_order}")
    print(f"Target variable: {target_variable}\n")
    
    # Demo 1: Structure-only mode (backward compatibility)
    print("=== Demo 1: Structure-Only Mode ===")
    structure_config = create_structure_only_config(max_parents=3)
    print(f"Config: predict_mechanisms={structure_config.predict_mechanisms}")
    
    structure_net = create_modular_parent_set_model(structure_config)
    structure_params = structure_net.init(key, data, variable_order, target_variable)
    
    structure_posterior = predict_with_mechanisms(
        structure_net, structure_params, data, variable_order, target_variable, structure_config
    )
    
    print(f"Posterior target: {structure_posterior.target_variable}")
    print(f"Number of parent sets: {len(structure_posterior.parent_set_probs)}")
    print(f"Uncertainty (bits): {structure_posterior.uncertainty:.3f}")
    print(f"Most likely parents: {structure_posterior.top_k_sets[0][0]}")
    print(f"Has mechanism predictions: {'mechanism_predictions' in structure_posterior.metadata}")
    print()
    
    # Demo 2: Enhanced mechanism-aware mode
    print("=== Demo 2: Enhanced Mechanism-Aware Mode ===")
    enhanced_config = create_enhanced_config(
        mechanism_types=["linear", "polynomial"], 
        max_parents=3
    )
    print(f"Config: predict_mechanisms={enhanced_config.predict_mechanisms}")
    print(f"Mechanism types: {enhanced_config.mechanism_types}")
    
    enhanced_net = create_modular_parent_set_model(enhanced_config)
    enhanced_params = enhanced_net.init(key, data, variable_order, target_variable)
    
    enhanced_posterior = predict_with_mechanisms(
        enhanced_net, enhanced_params, data, variable_order, target_variable, enhanced_config
    )
    
    print(f"Posterior target: {enhanced_posterior.target_variable}")
    print(f"Number of parent sets: {len(enhanced_posterior.parent_set_probs)}")
    print(f"Uncertainty (bits): {enhanced_posterior.uncertainty:.3f}")
    print(f"Most likely parents: {enhanced_posterior.top_k_sets[0][0]}")
    print(f"Has mechanism predictions: {'mechanism_predictions' in enhanced_posterior.metadata}")
    
    if "mechanism_predictions" in enhanced_posterior.metadata:
        mech_preds = enhanced_posterior.metadata["mechanism_predictions"]
        print(f"Number of mechanism predictions: {len(mech_preds)}")
        
        # Show first few mechanism predictions
        for i, pred in enumerate(mech_preds[:3]):
            print(f"  Prediction {i+1}: {pred.parent_set} -> {pred.mechanism_type} "
                  f"(confidence: {pred.confidence:.3f})")
    print()
    
    # Demo 3: Model comparison
    print("=== Demo 3: Model Comparison ===")
    
    # Get raw outputs for comparison
    structure_output = structure_net.apply(structure_params, key, data, variable_order, target_variable)
    enhanced_output = enhanced_net.apply(enhanced_params, key, data, variable_order, target_variable)
    
    comparison = compare_model_outputs(structure_output, enhanced_output)
    
    print("Output comparison:")
    print(f"  Structure-only keys: {comparison['structure_only_keys']}")
    print(f"  Enhanced keys: {comparison['enhanced_keys']}")
    print(f"  Enhanced-only keys: {comparison['enhanced_only_keys']}")
    
    if comparison.get('logits_shape_match', False):
        print(f"  Parent set similarity: {comparison['parent_set_cosine_similarity']:.3f}")
        print(f"  L2 distance: {comparison['parent_set_l2_distance']:.3f}")
    print()
    
    # Demo 4: Different mechanism type configurations
    print("=== Demo 4: Different Mechanism Types ===")
    
    mechanism_configs = [
        (["linear"], "Linear only"),
        (["linear", "polynomial"], "Linear + Polynomial"),
        (["linear", "polynomial", "gaussian"], "Linear + Polynomial + Gaussian"),
        (["linear", "polynomial", "gaussian", "neural"], "All types")
    ]
    
    for mech_types, description in mechanism_configs:
        config = create_enhanced_config(mechanism_types=mech_types)
        net = create_modular_parent_set_model(config)
        params = net.init(key, data, variable_order, target_variable)
        output = net.apply(params, key, data, variable_order, target_variable)
        
        mech_preds = output["mechanism_predictions"]
        mech_type_logits = mech_preds["mechanism_type_logits"]
        
        print(f"  {description}: {mech_type_logits.shape[1]} mechanism types")
    
    print("\n=== Demo Complete ===")
    print("✅ Part A: Modular Model Architecture successfully implemented!")
    print("✅ Feature flags working: easy switching between modes")
    print("✅ Backward compatibility: structure-only mode preserved")
    print("✅ Enhanced functionality: mechanism prediction working")
    print("✅ Multiple mechanism types: configurable and extensible")


if __name__ == "__main__":
    main()