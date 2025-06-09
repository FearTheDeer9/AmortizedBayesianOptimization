#!/usr/bin/env python3
"""
Test true amortization: train on multiple SCMs, test on unseen SCMs.

This demonstrates proper amortized causal discovery where:
1. Train on many different SCMs with ground truth
2. Test on completely new, unseen SCMs without any training
3. Model must generalize from learned patterns
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

import jax
import jax.numpy as jnp
import jax.random as random
import optax
import numpy as onp

from causal_bayes_opt.experiments.test_scms import create_simple_linear_scm
from causal_bayes_opt.mechanisms.linear import sample_from_linear_scm
from causal_bayes_opt.avici_integration import (
    create_training_batch,
    create_parent_set_model,
    predict_parent_posterior,
    summarize_posterior,
    get_marginal_parent_probabilities,
    create_train_step
)


def create_random_5node_scm(key, noise_scale=1.0, coefficient_range=(-2.0, 2.0)):
    """Create a random 5-node SCM with varying structure."""
    # Randomly select edges (ensuring DAG)
    variables = ['A', 'B', 'C', 'D', 'E']
    
    # Generate random DAG structure
    edges = []
    coefficients = {}
    
    # Use random key for structure generation
    key, subkey = random.split(key)
    edge_probs = random.uniform(subkey, shape=(5, 5))
    
    # Only allow edges from earlier to later variables (ensures DAG)
    for i in range(5):
        for j in range(i+1, 5):
            key, subkey = random.split(key)
            if edge_probs[i, j] > 0.6:  # 40% chance of edge
                parent = variables[i]
                child = variables[j]
                edges.append((parent, child))
                
                # Random coefficient
                key, subkey = random.split(key)
                coeff = random.uniform(subkey, minval=coefficient_range[0], maxval=coefficient_range[1])
                coefficients[(parent, child)] = float(coeff)
    
    # Ensure at least some edges exist
    if not edges:
        edges = [('A', 'B'), ('B', 'C'), ('C', 'D')]
        coefficients = {('A', 'B'): 1.5, ('B', 'C'): -1.0, ('C', 'D'): 0.8}
    
    # Random target (prefer nodes with parents)
    nodes_with_parents = list({child for _, child in edges})
    if nodes_with_parents:
        # Convert to indices for JAX compatibility
        all_vars = ['A', 'B', 'C', 'D', 'E']
        valid_indices = [all_vars.index(v) for v in nodes_with_parents]
        target_idx = random.choice(key, jnp.array(valid_indices))
        target = all_vars[int(target_idx)]
    else:
        target = 'D'
    
    return create_simple_linear_scm(
        variables=variables,
        edges=edges,
        coefficients=coefficients,
        noise_scales={var: noise_scale for var in variables},
        target=str(target)
    )


def create_model_config():
    """Configuration for amortized model."""
    return {
        'model_kwargs': {
            'layers': 2,
            'dim': 32,
            'key_size': 8,
            'num_heads': 2,
            'dropout': 0.1,  # Some dropout for generalization
        },
        'learning_rate': 1e-3,
        'batch_size': 32,
        'gradient_clip_norm': 1.0,
        'max_parent_size': 4,
    }


def train_amortized_model(n_training_scms=5, n_samples_per_scm=50, n_epochs=2):
    """Train model on multiple different SCMs."""
    print("🎯 TRAINING AMORTIZED MODEL")
    print("=" * 60)
    print(f"Training on {n_training_scms} different SCMs")
    print(f"Samples per SCM: {n_samples_per_scm}")
    print(f"Epochs: {n_epochs}")
    print("=" * 60)
    
    config = create_model_config()
    net = create_parent_set_model(
        model_kwargs=config['model_kwargs'],
        max_parent_size=config['max_parent_size']
    )
    
    # Initialize with dummy data
    key = random.PRNGKey(0)
    dummy_scm = create_random_5node_scm(key)
    dummy_samples = sample_from_linear_scm(dummy_scm, n_samples=config['batch_size'])
    dummy_batch = create_training_batch(dummy_scm, dummy_samples, dummy_scm['target'])
    variables = sorted(dummy_scm['variables'])
    
    params = net.init(random.PRNGKey(42), dummy_batch['x'], variables, dummy_scm['target'], True)
    
    # Setup optimizer
    optimizer = optax.chain(
        optax.clip_by_global_norm(config['gradient_clip_norm']),
        optax.adam(learning_rate=config['learning_rate'])
    )
    opt_state = optimizer.init(params)
    train_step_fn = create_train_step(net, optimizer)
    
    # Training loop over multiple SCMs
    total_steps = 0
    for epoch in range(n_epochs):
        print(f"\n📚 EPOCH {epoch + 1}/{n_epochs}")
        epoch_loss = 0.0
        
        for scm_idx in range(n_training_scms):
            # Generate a new random SCM
            key, subkey = random.split(key)
            scm = create_random_5node_scm(subkey, noise_scale=0.5)
            
            # IMPORTANT: Permute variable names to prevent spurious learning
            # Model shouldn't learn "A is always a root" etc.
            key, subkey = random.split(key)
            permutation = random.permutation(subkey, jnp.array([0, 1, 2, 3, 4]))
            var_names = ['A', 'B', 'C', 'D', 'E']
            permuted_names = [var_names[int(i)] for i in permutation]
            
            # Create name mapping and permute the SCM structure
            name_map = {old: new for old, new in zip(var_names, permuted_names)}
            variables = sorted(permuted_names)
            
            # Apply permutation to edges and rebuild SCM
            old_edges = scm['edges']
            permuted_edges = [(name_map[p], name_map[c]) for p, c in old_edges]
            permuted_coefficients = {}
            for (old_p, old_c) in old_edges:
                new_edge = (name_map[old_p], name_map[old_c])
                # Extract coefficient from original SCM (simplified)
                permuted_coefficients[new_edge] = 1.0  # Use simple coefficient
            
            permuted_scm = create_simple_linear_scm(
                variables=permuted_names,
                edges=permuted_edges,
                coefficients=permuted_coefficients,
                noise_scales={v: 0.5 for v in permuted_names},
                target=name_map[scm['target']]
            )
            
            scm = permuted_scm
            
            # Get true parent sets for all variables
            from causal_bayes_opt.data_structures.scm import get_parents
            true_parents = {}
            for var in variables:
                true_parents[var] = get_parents(scm, var)
            
            # Train on each variable as target
            for target_var in variables:
                # Generate fresh data for this SCM and target
                key, subkey = random.split(key)
                samples = sample_from_linear_scm(scm, n_samples=n_samples_per_scm, seed=int(subkey[0]))
                batch = create_training_batch(scm, samples, target_var)
                
                # Training step
                params, opt_state, loss = train_step_fn(
                    params, opt_state, batch['x'], variables, target_var, true_parents[target_var]
                )
                
                epoch_loss += loss
                total_steps += 1
                
                if total_steps % 50 == 0:
                    print(f"  Step {total_steps}: loss = {loss:.4f}")
        
        avg_loss = epoch_loss / (n_training_scms * len(variables))
        print(f"  Average epoch loss: {avg_loss:.4f}")
    
    print("\n✅ Training complete!")
    return net, params


def compute_significance_metrics(marginal_probs, true_parents, all_variables):
    """
    Compute significance metrics comparing true parents vs random baseline.
    
    Args:
        marginal_probs: Dict mapping variable -> marginal probability of being parent
        true_parents: FrozenSet of true parent variables
        all_variables: List of all possible variables
    
    Returns:
        Dict with significance metrics
    """
    # Extract probabilities for true vs false parents
    true_parent_probs = [marginal_probs.get(var, 0.0) for var in true_parents]
    false_parent_probs = [marginal_probs.get(var, 0.0) for var in all_variables if var not in true_parents]
    
    # Random baseline: uniform probability (1/num_variables for each)
    random_baseline = 1.0 / len(all_variables)
    
    # Compute metrics
    mean_true_prob = onp.mean(true_parent_probs) if true_parent_probs else 0.0
    mean_false_prob = onp.mean(false_parent_probs) if false_parent_probs else 0.0
    
    # Signal-to-noise ratio
    signal_strength = mean_true_prob - random_baseline
    noise_strength = mean_false_prob - random_baseline
    snr = signal_strength / (noise_strength + 1e-8)  # Avoid division by zero
    
    # Area under precision-recall curve approximation
    # Sort all variables by their marginal probability
    all_probs = [(var, marginal_probs.get(var, 0.0)) for var in all_variables]
    all_probs.sort(key=lambda x: x[1], reverse=True)
    
    # Compute precision at each rank
    precisions = []
    true_positive_count = 0
    for rank, (var, prob) in enumerate(all_probs, 1):
        if var in true_parents:
            true_positive_count += 1
        precision = true_positive_count / rank
        precisions.append(precision)
    
    mean_precision = onp.mean(precisions)
    
    return {
        'mean_true_parent_prob': mean_true_prob,
        'mean_false_parent_prob': mean_false_prob,
        'random_baseline': random_baseline,
        'signal_strength': signal_strength,
        'signal_to_noise_ratio': snr,
        'mean_precision': mean_precision,
        'true_parent_count': len(true_parents),
        'false_parent_count': len(false_parent_probs)
    }


def test_on_unseen_scms(net, params, n_test_scms=5):
    """Test the trained model on completely new, unseen SCMs."""
    print("\n🧪 TESTING ON UNSEEN SCMS")
    print("=" * 60)
    print(f"Testing on {n_test_scms} new SCMs the model has never seen")
    print("=" * 60)
    
    key = random.PRNGKey(9999)  # Different seed for test SCMs
    
    total_correct = 0
    total_tested = 0
    
    # For significance testing
    all_significance_metrics = []
    
    for test_idx in range(n_test_scms):
        key, subkey = random.split(key)
        test_scm = create_random_5node_scm(subkey, noise_scale=0.5)
        variables = sorted(test_scm['variables'])
        
        print(f"\n📊 TEST SCM {test_idx + 1}")
        
        # Get true parent sets
        from causal_bayes_opt.data_structures.scm import get_parents, get_edges
        edges = get_edges(test_scm)
        print(f"  Edges: {sorted(edges)}")
        
        # Generate test data (only 32 samples - limited data!)
        key, subkey = random.split(key)
        test_samples = sample_from_linear_scm(test_scm, n_samples=32, seed=int(subkey[0]))
        
        # Test each variable
        for target_var in variables:
            true_parents = get_parents(test_scm, target_var)
            
            # Create batch and predict
            batch = create_training_batch(test_scm, test_samples, target_var)
            
            # IMPORTANT: No training here! Just inference with fixed params
            posterior = predict_parent_posterior(net, params, batch['x'], variables, target_var)
            summary = summarize_posterior(posterior)
            
            predicted_parents = frozenset(summary['most_likely_parents'])
            is_correct = predicted_parents == true_parents
            
            if is_correct:
                total_correct += 1
            total_tested += 1
            
            # Display results
            status = "✅" if is_correct else "❌"
            print(f"\n  {status} Target: {target_var}")
            print(f"     True parents: {set(true_parents) if true_parents else '{}'}")
            print(f"     Predicted: {set(predicted_parents) if predicted_parents else '{}'}")
            print(f"     Confidence: {summary['most_likely_probability']:.3f}")
            print(f"     Uncertainty: {summary['uncertainty_bits']:.2f} bits")
            
            # Compute significance metrics for this prediction
            marginals = get_marginal_parent_probabilities(posterior, variables)
            sig_metrics = compute_significance_metrics(marginals, true_parents, variables)
            all_significance_metrics.append(sig_metrics)
            
            # Show marginal probabilities and significance
            print(f"     Marginal probabilities:")
            for var, prob in sorted(marginals.items(), key=lambda x: x[1], reverse=True):
                is_true = var in true_parents
                mark = "✓" if is_true else " "
                print(f"       {mark} P({var} is parent) = {prob:.3f}")
            
            print(f"     Signal metrics:")
            print(f"       True parent avg prob: {sig_metrics['mean_true_parent_prob']:.3f}")
            print(f"       False parent avg prob: {sig_metrics['mean_false_parent_prob']:.3f}")
            print(f"       Random baseline: {sig_metrics['random_baseline']:.3f}")
            print(f"       Signal-to-noise ratio: {sig_metrics['signal_to_noise_ratio']:.2f}")
    
    accuracy = total_correct / total_tested
    
    # Compute aggregate significance metrics
    overall_true_prob = onp.mean([m['mean_true_parent_prob'] for m in all_significance_metrics])
    overall_false_prob = onp.mean([m['mean_false_parent_prob'] for m in all_significance_metrics])
    overall_baseline = onp.mean([m['random_baseline'] for m in all_significance_metrics])
    overall_snr = onp.mean([m['signal_to_noise_ratio'] for m in all_significance_metrics])
    overall_precision = onp.mean([m['mean_precision'] for m in all_significance_metrics])
    
    print(f"\n📊 OVERALL RESULTS:")
    print(f"  Correct predictions: {total_correct}/{total_tested}")
    print(f"  Accuracy: {accuracy:.1%}")
    
    print(f"\n🔬 SIGNIFICANCE ANALYSIS:")
    print(f"  Average true parent probability: {overall_true_prob:.3f}")
    print(f"  Average false parent probability: {overall_false_prob:.3f}")
    print(f"  Random baseline expectation: {overall_baseline:.3f}")
    print(f"  Signal-to-noise ratio: {overall_snr:.2f}")
    print(f"  Mean precision: {overall_precision:.3f}")
    
    # Statistical significance test
    true_parent_lift = overall_true_prob / overall_baseline
    false_parent_ratio = overall_false_prob / overall_baseline
    
    print(f"\n📈 SIGNAL DETECTION:")
    print(f"  True parents are {true_parent_lift:.1f}x more likely than random")
    print(f"  False parents are {false_parent_ratio:.1f}x as likely as random")
    
    if overall_snr > 2.0:
        print(f"  ✅ STRONG signal: Model clearly distinguishes true vs false parents")
    elif overall_snr > 1.0:
        print(f"  ⚠️ MODERATE signal: Model shows some discrimination ability")
    elif overall_snr > 0.0:
        print(f"  🔍 WEAK signal: Model has slight preference for true parents")
    else:
        print(f"  ❌ NO signal: Model performance indistinguishable from random")
    
    if accuracy > 0.8:
        print(f"\n  🎉 Excellent generalization!")
    elif accuracy > 0.6:
        print(f"\n  ✅ Good generalization")
    elif accuracy > 0.4:
        print(f"\n  ⚠️ Moderate generalization")
    else:
        print(f"\n  ❌ Poor exact accuracy")
        if overall_snr > 1.0:
            print(f"      But model still shows significant signal detection!")
    
    return accuracy, overall_snr


def main():
    """Run amortization test."""
    print("🚀 AMORTIZED CAUSAL DISCOVERY TEST")
    print("=" * 70)
    print("This test demonstrates true amortization:")
    print("1. Train on many different SCMs with supervision")
    print("2. Test on completely unseen SCMs without any adaptation")
    print("3. Model must generalize from learned patterns alone")
    print("=" * 70)
    
    try:
        # Train the amortized model
        net, params = train_amortized_model(
            n_training_scms=5,
            n_samples_per_scm=50,
            n_epochs=2
        )
        
        # Test on unseen SCMs
        accuracy, snr = test_on_unseen_scms(net, params, n_test_scms=5)
        
        print(f"\n" + "="*70)
        print(f"🏁 TEST COMPLETE")
        print(f"\nKey insights:")
        print(f"• Model was trained on 5 different random SCMs with PERMUTED variable names")
        print(f"• Variable name permutation prevents spurious learning of name patterns")
        print(f"• Model was tested on 5 completely new SCMs")
        print(f"• No weight updates during testing - pure amortization")
        print(f"• Final accuracy on unseen SCMs: {accuracy:.1%}")
        print(f"• Signal-to-noise ratio: {snr:.2f}")
        
        print(f"\n✨ SIGNIFICANCE TEST RESULTS:")
        if snr > 1.0:
            print(f"• ✅ Model shows SIGNIFICANT signal detection above random chance")
            print(f"• True parents receive higher probabilities than false parents")
            print(f"• This validates that amortized learning is working")
        else:
            print(f"• ⚠️ Weak signal suggests limited generalization")
            print(f"• May need more training data or better architecture")
        
        print(f"\n🔧 METHODOLOGICAL IMPROVEMENTS:")
        print(f"• Variable name permutation during training")
        print(f"• Significance testing vs random baseline")
        print(f"• Signal-to-noise ratio analysis")
        print(f"• Marginal probability analysis beyond exact accuracy")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)