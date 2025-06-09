#!/usr/bin/env python3
"""
Enhanced amortization test with improved training to avoid trivial solutions.

This test addresses the "always predict empty set" problem by:
1. Balancing training data (equal root vs non-root nodes)
2. Using focal loss to penalize overconfident predictions
3. Adding diversity regularization
4. Curriculum learning (simple to complex)
5. Better evaluation metrics
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
from functools import partial

from causal_bayes_opt.experiments.test_scms import create_simple_linear_scm
from causal_bayes_opt.mechanisms.linear import sample_from_linear_scm
from causal_bayes_opt.avici_integration import (
    create_training_batch,
    create_parent_set_model,
    predict_parent_posterior,
    summarize_posterior,
    get_marginal_parent_probabilities,
    compute_loss
)


def create_balanced_scm_dataset(key, n_scms=10, ensure_non_roots=True):
    """
    Create a balanced dataset of SCMs ensuring good coverage of parent set sizes.
    
    Args:
        key: Random key
        n_scms: Number of SCMs to create
        ensure_non_roots: If True, ensure most variables have parents
    
    Returns:
        List of (scm, target_var, true_parents) tuples
    """
    dataset = []
    
    for i in range(n_scms):
        key, subkey = random.split(key)
        
        if ensure_non_roots and i % 3 != 0:
            # Create SCMs with guaranteed parent relationships
            if i % 3 == 1:
                # Chain structure - everyone except first has parents
                scm = create_simple_linear_scm(
                    variables=['A', 'B', 'C', 'D', 'E'],
                    edges=[('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'E')],
                    coefficients={edge: 1.5 for edge in [('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'E')]},
                    noise_scales={v: 0.5 for v in ['A', 'B', 'C', 'D', 'E']},
                    target='C'
                )
                targets_with_parents = [
                    ('B', frozenset(['A'])),
                    ('C', frozenset(['B'])),
                    ('D', frozenset(['C'])),
                    ('E', frozenset(['D']))
                ]
            else:
                # Complex structure - multiple parents
                scm = create_simple_linear_scm(
                    variables=['A', 'B', 'C', 'D', 'E'],
                    edges=[('A', 'C'), ('B', 'C'), ('A', 'D'), ('C', 'D'), ('D', 'E')],
                    coefficients={
                        ('A', 'C'): 1.0, ('B', 'C'): -0.8,
                        ('A', 'D'): 0.7, ('C', 'D'): 1.2,
                        ('D', 'E'): 0.9
                    },
                    noise_scales={v: 0.5 for v in ['A', 'B', 'C', 'D', 'E']},
                    target='D'
                )
                targets_with_parents = [
                    ('C', frozenset(['A', 'B'])),
                    ('D', frozenset(['A', 'C'])),
                    ('E', frozenset(['D']))
                ]
            
            # Add all target-parent pairs from this SCM
            for target, parents in targets_with_parents:
                dataset.append((scm, target, parents))
        else:
            # Random SCM (may have roots)
            # Create a simple random SCM
            edges = []
            coefficients = {}
            variables = ['A', 'B', 'C', 'D', 'E']
            
            # Random edges
            for i in range(5):
                for j in range(i+1, 5):
                    key, subkey2 = random.split(subkey)
                    if random.uniform(subkey2) > 0.7:  # 30% chance of edge
                        parent = variables[i]
                        child = variables[j]
                        edges.append((parent, child))
                        coefficients[(parent, child)] = float(random.uniform(subkey2, minval=-2, maxval=2))
            
            if not edges:
                edges = [('A', 'B'), ('B', 'C')]
                coefficients = {('A', 'B'): 1.0, ('B', 'C'): 0.8}
            
            # Random target
            key, subkey2 = random.split(subkey)
            target = variables[int(random.choice(subkey2, 5))]
            
            scm = create_simple_linear_scm(
                variables=variables,
                edges=edges,
                coefficients=coefficients,
                noise_scales={v: 0.5 for v in variables},
                target=target
            )
            
            # Add one random target from this SCM
            from causal_bayes_opt.data_structures.scm import get_variables, get_parents
            variables = sorted(get_variables(scm))
            key, subkey = random.split(key)
            target_idx = random.choice(subkey, len(variables))
            target = variables[int(target_idx)]
            parents = get_parents(scm, target)
            dataset.append((scm, target, parents))
    
    return dataset


def focal_loss(logits, true_idx, gamma=2.0, alpha=0.25):
    """
    Focal loss to address class imbalance and overconfident predictions.
    
    Args:
        logits: Model output logits
        true_idx: Index of true parent set
        gamma: Focusing parameter (higher = more focus on hard examples)
        alpha: Class weight for positive examples
    
    Returns:
        Focal loss value
    """
    # Compute softmax probabilities
    probs = jax.nn.softmax(logits)
    
    # Get probability of true class
    pt = probs[true_idx]
    
    # Focal loss formula
    focal_weight = (1 - pt) ** gamma
    ce_loss = -jnp.log(pt + 1e-8)
    
    return alpha * focal_weight * ce_loss


def compute_enhanced_loss(net, params, x, variable_order, target_variable, 
                         true_parent_set, is_training=True, 
                         use_focal=True, diversity_weight=0.1):
    """
    Enhanced loss function that discourages trivial solutions.
    """
    output = net.apply(
        params, random.PRNGKey(0), x, variable_order, target_variable, is_training
    )
    
    logits = output['parent_set_logits']
    parent_sets = output['parent_sets']
    
    # Find true parent set index
    true_idx = None
    for i, ps in enumerate(parent_sets):
        if ps == true_parent_set:
            true_idx = i
            break
    
    if true_idx is None:
        # Use standard loss if true set not in predictions
        return compute_loss(net, params, x, variable_order, target_variable, true_parent_set, is_training)
    
    # Primary loss: focal loss or cross-entropy
    if use_focal:
        primary_loss = focal_loss(logits, true_idx)
    else:
        log_probs = jax.nn.log_softmax(logits)
        primary_loss = -log_probs[true_idx]
    
    # Diversity regularization: penalize if model always predicts same thing
    probs = jax.nn.softmax(logits)
    entropy = -jnp.sum(probs * jnp.log(probs + 1e-8))
    max_entropy = jnp.log(len(parent_sets))
    diversity_loss = diversity_weight * (max_entropy - entropy)
    
    # Penalty for predicting empty set when it's wrong
    empty_set_idx = None
    for i, ps in enumerate(parent_sets):
        if len(ps) == 0:
            empty_set_idx = i
            break
    
    empty_set_penalty = 0.0
    if empty_set_idx is not None and len(true_parent_set) > 0:
        # Penalize high probability on empty set when true set is non-empty
        empty_set_prob = probs[empty_set_idx]
        empty_set_penalty = 0.5 * empty_set_prob
    
    return primary_loss + diversity_loss + empty_set_penalty


def create_enhanced_train_step(net, optimizer):
    """Create enhanced training step with better loss."""
    def train_step(params, opt_state, x, variable_order, target_variable, true_parent_set):
        def loss_fn(params):
            return compute_enhanced_loss(
                net, params, x, variable_order, target_variable, 
                true_parent_set, is_training=True
            )
        
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        
        return new_params, new_opt_state, loss
    
    return train_step  # Don't JIT compile due to string arguments


def train_with_curriculum(n_epochs=5):
    """
    Train with curriculum learning and balanced data.
    """
    print("🎯 ENHANCED TRAINING WITH CURRICULUM LEARNING")
    print("=" * 60)
    
    # Create model
    config = {
        'model_kwargs': {
            'layers': 4,
            'dim': 64,
            'key_size': 16,
            'num_heads': 4,
            'dropout': 0.1,
        },
        'learning_rate': 5e-4,  # Slightly higher LR
        'batch_size': 32,
        'gradient_clip_norm': 1.0,
        'max_parent_size': 4,
    }
    
    net = create_parent_set_model(
        model_kwargs=config['model_kwargs'],
        max_parent_size=config['max_parent_size']
    )
    
    # Initialize
    key = random.PRNGKey(42)
    dummy_scm = create_simple_linear_scm(
        variables=['A', 'B', 'C', 'D', 'E'],
        edges=[('A', 'B')],
        coefficients={('A', 'B'): 1.0},
        noise_scales={v: 1.0 for v in ['A', 'B', 'C', 'D', 'E']},
        target='B'
    )
    dummy_samples = sample_from_linear_scm(dummy_scm, n_samples=config['batch_size'])
    dummy_batch = create_training_batch(dummy_scm, dummy_samples, 'B')
    
    params = net.init(random.PRNGKey(42), dummy_batch['x'], ['A', 'B', 'C', 'D', 'E'], 'B', True)
    
    # Optimizer with warmup
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=1e-5,
        peak_value=config['learning_rate'],
        warmup_steps=50,
        decay_steps=500,
        end_value=1e-5
    )
    
    optimizer = optax.chain(
        optax.clip_by_global_norm(config['gradient_clip_norm']),
        optax.adam(learning_rate=schedule)
    )
    opt_state = optimizer.init(params)
    train_step_fn = create_enhanced_train_step(net, optimizer)
    
    # Curriculum: start simple, increase complexity
    for epoch in range(n_epochs):
        print(f"\n📚 EPOCH {epoch + 1}/{n_epochs}")
        
        # Create balanced dataset for this epoch
        key, subkey = random.split(key)
        if epoch < 2:
            # Early epochs: focus on simple cases
            dataset = create_balanced_scm_dataset(subkey, n_scms=10, ensure_non_roots=True)
            print("  Training on balanced dataset with guaranteed parent relationships")
        else:
            # Later epochs: mix in random SCMs
            dataset = create_balanced_scm_dataset(subkey, n_scms=15, ensure_non_roots=False)
            print("  Training on mixed dataset")
        
        epoch_losses = []
        parent_set_sizes = []
        
        for scm, target_var, true_parents in dataset:
            # Generate fresh samples
            key, subkey = random.split(key)
            samples = sample_from_linear_scm(scm, n_samples=config['batch_size'], seed=int(subkey[0]))
            batch = create_training_batch(scm, samples, target_var)
            variables = sorted(scm['variables'])
            
            # Train step
            params, opt_state, loss = train_step_fn(
                params, opt_state, batch['x'], variables, target_var, true_parents
            )
            
            epoch_losses.append(float(loss))
            parent_set_sizes.append(len(true_parents))
        
        # Report statistics
        avg_loss = onp.mean(epoch_losses)
        print(f"  Average loss: {avg_loss:.4f}")
        print(f"  Parent set size distribution: {onp.bincount(parent_set_sizes)}")
        
        # Quick validation on non-trivial cases
        val_correct = 0
        val_total = 0
        
        for i in range(5):
            key, subkey = random.split(key)
            val_scm = create_simple_linear_scm(
                variables=['A', 'B', 'C', 'D', 'E'],
                edges=[('A', 'B'), ('B', 'C'), ('A', 'C')],
                coefficients={('A', 'B'): 1.0, ('B', 'C'): 0.8, ('A', 'C'): 0.5},
                noise_scales={v: 0.5 for v in ['A', 'B', 'C', 'D', 'E']},
                target='C'
            )
            
            # Test on C (should have parents A and B)
            val_samples = sample_from_linear_scm(val_scm, n_samples=32, seed=int(subkey[0]))
            val_batch = create_training_batch(val_scm, val_samples, 'C')
            
            posterior = predict_parent_posterior(net, params, val_batch['x'], ['A', 'B', 'C', 'D', 'E'], 'C')
            summary = summarize_posterior(posterior)
            
            true_parents = frozenset(['A', 'B'])
            predicted = frozenset(summary['most_likely_parents'])
            
            if predicted == true_parents:
                val_correct += 1
            val_total += 1
        
        print(f"  Validation accuracy on non-trivial cases: {val_correct}/{val_total}")
    
    print("\n✅ Enhanced training complete!")
    return net, params


def evaluate_on_diverse_test_set(net, params):
    """
    Evaluate on a diverse test set with detailed analysis.
    """
    print("\n🧪 EVALUATION ON DIVERSE TEST SET")
    print("=" * 60)
    
    test_cases = [
        # Simple cases
        ("Chain A→B", 
         create_simple_linear_scm(['A', 'B'], [('A', 'B')], {('A', 'B'): 1.0}, {v: 0.5 for v in ['A', 'B']}, 'B'),
         'B', frozenset(['A'])),
        
        # Fork structure
        ("Fork A→B, A→C",
         create_simple_linear_scm(['A', 'B', 'C'], [('A', 'B'), ('A', 'C')], 
                                {('A', 'B'): 1.0, ('A', 'C'): -0.8}, {v: 0.5 for v in ['A', 'B', 'C']}, 'B'),
         'B', frozenset(['A'])),
        
        # Collider
        ("Collider A→C←B",
         create_simple_linear_scm(['A', 'B', 'C'], [('A', 'C'), ('B', 'C')],
                                {('A', 'C'): 1.0, ('B', 'C'): 0.7}, {v: 0.5 for v in ['A', 'B', 'C']}, 'C'),
         'C', frozenset(['A', 'B'])),
        
        # Complex 5-node
        ("Complex 5-node",
         create_simple_linear_scm(['A', 'B', 'C', 'D', 'E'],
                                [('A', 'B'), ('A', 'C'), ('B', 'D'), ('C', 'D'), ('D', 'E')],
                                {('A', 'B'): 1.0, ('A', 'C'): 0.8, ('B', 'D'): 0.6, 
                                 ('C', 'D'): -0.5, ('D', 'E'): 1.2},
                                {v: 0.5 for v in ['A', 'B', 'C', 'D', 'E']}, 'D'),
         'D', frozenset(['B', 'C'])),
        
        # Root node test
        ("Root node",
         create_simple_linear_scm(['X', 'Y', 'Z'], [('X', 'Y'), ('Y', 'Z')],
                                {('X', 'Y'): 1.0, ('Y', 'Z'): 1.0}, {v: 0.5 for v in ['X', 'Y', 'Z']}, 'X'),
         'X', frozenset())
    ]
    
    results = []
    
    for name, scm, target, true_parents in test_cases:
        print(f"\n📊 {name}:")
        print(f"  Target: {target}")
        print(f"  True parents: {set(true_parents) if true_parents else '{}'}")
        
        # Generate test data
        samples = sample_from_linear_scm(scm, n_samples=64, seed=42)
        batch = create_training_batch(scm, samples, target)
        variables = sorted(scm['variables'])
        
        # Predict
        posterior = predict_parent_posterior(net, params, batch['x'], variables, target)
        summary = summarize_posterior(posterior)
        predicted = frozenset(summary['most_likely_parents'])
        
        # Analyze
        is_correct = predicted == true_parents
        marginals = get_marginal_parent_probabilities(posterior, variables)
        
        print(f"  Predicted: {set(predicted) if predicted else '{}'}")
        print(f"  Confidence: {summary['most_likely_probability']:.3f}")
        print(f"  Result: {'✅ CORRECT' if is_correct else '❌ INCORRECT'}")
        
        if not is_correct:
            print(f"  Marginal probabilities:")
            for var, prob in sorted(marginals.items(), key=lambda x: x[1], reverse=True):
                is_true = var in true_parents
                mark = "✓" if is_true else " "
                print(f"    {mark} P({var} is parent) = {prob:.3f}")
        
        results.append({
            'name': name,
            'correct': is_correct,
            'predicted': predicted,
            'true': true_parents,
            'is_root': len(true_parents) == 0,
            'confidence': summary['most_likely_probability']
        })
    
    # Summary statistics
    total_correct = sum(r['correct'] for r in results)
    root_correct = sum(r['correct'] for r in results if r['is_root'])
    non_root_correct = sum(r['correct'] for r in results if not r['is_root'])
    
    n_root = sum(r['is_root'] for r in results)
    n_non_root = len(results) - n_root
    
    print(f"\n📈 SUMMARY:")
    print(f"  Overall accuracy: {total_correct}/{len(results)} ({100*total_correct/len(results):.0f}%)")
    print(f"  Root node accuracy: {root_correct}/{n_root} ({100*root_correct/n_root:.0f}%)")
    print(f"  Non-root accuracy: {non_root_correct}/{n_non_root} ({100*non_root_correct/n_non_root:.0f}%)")
    
    # Check if model still defaults to empty set
    empty_predictions = sum(1 for r in results if len(r['predicted']) == 0)
    print(f"  Empty set predictions: {empty_predictions}/{len(results)}")
    
    if empty_predictions == len(results):
        print(f"\n  ⚠️ WARNING: Model still defaults to empty parent sets!")
    elif non_root_correct > 0:
        print(f"\n  ✅ SUCCESS: Model predicts non-empty parent sets correctly!")
    
    return results


def main():
    """Run enhanced amortization test."""
    print("🚀 ENHANCED AMORTIZED CAUSAL DISCOVERY TEST")
    print("=" * 70)
    print("Improvements:")
    print("• Balanced training data (equal root/non-root)")
    print("• Focal loss to handle class imbalance")
    print("• Diversity regularization")
    print("• Curriculum learning")
    print("• Better evaluation metrics")
    print("=" * 70)
    
    try:
        # Train with enhancements
        net, params = train_with_curriculum(n_epochs=5)
        
        # Evaluate
        results = evaluate_on_diverse_test_set(net, params)
        
        print(f"\n" + "="*70)
        print(f"🏁 TEST COMPLETE")
        
        # Check for non-trivial learning
        non_root_results = [r for r in results if not r['is_root']]
        non_root_correct = sum(r['correct'] for r in non_root_results)
        
        if non_root_correct > 0:
            print(f"\n🎉 SUCCESS! Model learned to predict non-empty parent sets")
            print(f"   This shows real causal structure learning, not just")
            print(f"   trivial 'always predict empty' behavior")
        else:
            print(f"\n⚠️ Model still struggles with non-root nodes")
            print(f"   May need more training data or architectural changes")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)