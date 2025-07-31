#!/usr/bin/env python3
"""Extract and compare marginal probabilities across different policies."""

import re
import sys

def extract_marginals(log_file):
    """Extract marginal probabilities for each policy-surrogate pair."""
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Find all policy+surrogate pairs
    pairs = re.findall(r'Evaluating pair: (\w+\+\w+)', content)
    
    results = {}
    
    for pair in pairs:
        # Extract section for this pair
        pattern = f'Evaluating pair: {re.escape(pair)}(.*?)(?=Evaluating pair:|Results saved to|$)'
        section = re.search(pattern, content, re.DOTALL)
        
        if not section:
            continue
            
        section_text = section.group(1)
        
        # Extract SCM evaluations
        scm_pattern = r'Testing on (\w+)\.\.\..*?target=\'(\w+)\', true_parents=({[^}]+})'
        scm_matches = re.findall(scm_pattern, section_text, re.DOTALL)
        
        # Extract marginal probabilities for each step
        marginal_pattern = r'Step (\d+) - Detailed marginal probabilities:.*?Method: (\w+)(.*?)(?=Step \d+ Summary:|$)'
        marginal_matches = re.findall(marginal_pattern, section_text, re.DOTALL)
        
        # Extract interventions
        intervention_pattern = r'Step (\d+) Summary:.*?Intervention: \[([^\]]+)\] = \[([^\]]+)\]'
        intervention_matches = re.findall(intervention_pattern, section_text)
        
        # Extract final F1/SHD
        f1_pattern = r'Final F1 score: ([\d.]+)'
        shd_pattern = r'Final SHD: ([\d.]+)'
        
        pair_results = {
            'scms': {},
            'interventions': intervention_matches,
            'marginals': []
        }
        
        # Parse marginals
        for step, method, marginal_text in marginal_matches:
            vars_probs = {}
            var_pattern = r'(\w+): ([\d.]+) \(actual parent: (\w+)\)'
            for var, prob, is_parent in re.findall(var_pattern, marginal_text):
                vars_probs[var] = {
                    'prob': float(prob),
                    'is_parent': is_parent == 'True'
                }
            pair_results['marginals'].append({
                'step': int(step),
                'method': method,
                'probabilities': vars_probs
            })
        
        # Extract final metrics per SCM
        for scm_name, target, true_parents in scm_matches:
            scm_section_pattern = f'Testing on {scm_name}(.*?)(?=Testing on|Aggregate metrics:|$)'
            scm_section = re.search(scm_section_pattern, section_text, re.DOTALL)
            if scm_section:
                scm_text = scm_section.group(1)
                f1_match = re.search(f1_pattern, scm_text)
                shd_match = re.search(shd_pattern, scm_text)
                
                pair_results['scms'][scm_name] = {
                    'target': target,
                    'true_parents': eval(true_parents),
                    'f1': float(f1_match.group(1)) if f1_match else None,
                    'shd': float(shd_match.group(1)) if shd_match else None
                }
        
        results[pair] = pair_results
    
    return results

def compare_marginals(results):
    """Compare marginal probabilities across different policies."""
    
    print("\n" + "="*80)
    print("MARGINAL PROBABILITY COMPARISON")
    print("="*80)
    
    # Group by SCM
    scms = set()
    for pair_data in results.values():
        scms.update(pair_data['scms'].keys())
    
    for scm in sorted(scms):
        print(f"\n\nSCM: {scm}")
        print("-" * 60)
        
        # Get true parents for this SCM
        true_parents = None
        target = None
        for pair_data in results.values():
            if scm in pair_data['scms']:
                true_parents = pair_data['scms'][scm]['true_parents']
                target = pair_data['scms'][scm]['target']
                break
        
        print(f"Target: {target}, True parents: {true_parents}")
        
        # Compare marginals across policies
        for pair, pair_data in sorted(results.items()):
            if scm not in pair_data['scms']:
                continue
                
            print(f"\n{pair}:")
            print(f"  Final F1: {pair_data['scms'][scm]['f1']}")
            print(f"  Final SHD: {pair_data['scms'][scm]['shd']}")
            
            # Show marginals for each step
            scm_marginals = [m for m in pair_data['marginals'] if any(var in m['probabilities'] for var in true_parents)]
            
            if scm_marginals:
                print("  Marginals by step:")
                for marginal in scm_marginals[:3]:  # Show first 3 steps
                    step = marginal['step']
                    print(f"    Step {step}:")
                    for var in sorted(marginal['probabilities'].keys()):
                        prob_data = marginal['probabilities'][var]
                        if var != target:  # Don't show target
                            parent_marker = "✓" if prob_data['is_parent'] else " "
                            print(f"      {var}: {prob_data['prob']:.3f} {parent_marker}")
    
    # Check if all policies have same F1/SHD
    print("\n\n" + "="*60)
    print("F1/SHD UNIFORMITY CHECK")
    print("="*60)
    
    for scm in sorted(scms):
        f1_scores = []
        shd_scores = []
        
        for pair, pair_data in results.items():
            if scm in pair_data['scms']:
                f1_scores.append((pair, pair_data['scms'][scm]['f1']))
                shd_scores.append((pair, pair_data['scms'][scm]['shd']))
        
        print(f"\n{scm}:")
        print("  F1 scores:", [(p, f) for p, f in f1_scores])
        print("  SHD scores:", [(p, s) for p, s in shd_scores])
        
        # Check if all the same
        unique_f1 = len(set(f for _, f in f1_scores))
        unique_shd = len(set(s for _, s in shd_scores))
        
        if unique_f1 == 1:
            print("  ⚠️  All policies have IDENTICAL F1 score!")
        else:
            print("  ✓ Policies have different F1 scores")
            
        if unique_shd == 1:
            print("  ⚠️  All policies have IDENTICAL SHD score!")
        else:
            print("  ✓ Policies have different SHD scores")

if __name__ == "__main__":
    log_file = sys.argv[1] if len(sys.argv) > 1 else "marginals_comparison.log"
    results = extract_marginals(log_file)
    compare_marginals(results)