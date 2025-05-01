"""Unit tests for data sampling utilities."""

import unittest
import pandas as pd
import numpy as np

# Assuming necessary imports from causal_meta will be added here later
from causal_meta.environments.scm import StructuralCausalModel
from causal_meta.environments.interventions import PerfectIntervention, ImperfectIntervention, SoftIntervention
from causal_meta.graph import CausalGraph
from causal_meta.environments.samplers import (
    sample_observational,
    sample_interventional,
    sample_counterfactual
)

class TestSamplers(unittest.TestCase):

    def setUp(self):
        """Set up a simple SCM for testing."""
        # Placeholder SCM setup - will be refined
        self.variables = ['X', 'Y', 'Z']
        self.scm = self._create_simple_scm()
        # pass # Removed pass

    def _create_simple_scm(self) -> StructuralCausalModel:
        """Helper to create a consistent simple SCM: X -> Y -> Z"""
        # Initialize graph and add nodes/edges separately
        graph = CausalGraph()
        nodes = ['X', 'Y', 'Z']
        edges = [('X', 'Y'), ('Y', 'Z')]

        # Add nodes individually
        for node in nodes:
            graph.add_node(node)

        # Add edges individually
        for u, v in edges:
            graph.add_edge(u, v)

        scm = StructuralCausalModel(graph)
        # Explicitly set variable names in SCM instance
        scm._variable_names = nodes

        # Define simple linear mechanisms with Gaussian noise
        # X = N(0, 1)
        scm.define_structural_equation('X', lambda noise: noise,
                                     exogenous_function=lambda: np.random.normal(0, 1))
        # Y = 2*X + N(0, 0.5)
        scm.define_structural_equation('Y', lambda X, noise: 2 * X + noise,
                                     exogenous_function=lambda: np.random.normal(0, 0.5))
        # Z = -1*Y + N(0, 0.2)
        scm.define_structural_equation('Z', lambda Y, noise: -1 * Y + noise,
                                     exogenous_function=lambda: np.random.normal(0, 0.2))
        return scm
        # raise NotImplementedError("SCM setup needs implementation") # Removed error

    # --- Observational Sampling Tests ---

    # @unittest.skip("Needs sampler implementation") # Unskipped
    def test_sample_observational_basic(self):
        """Test basic observational sampling returns correct format and size."""
        samples = sample_observational(self.scm, num_samples=100)
        self.assertIsInstance(samples, pd.DataFrame)
        self.assertEqual(samples.shape[0], 100)
        self.assertListEqual(list(samples.columns), self.variables)
        # pass # Removed pass

    # @unittest.skip("Needs sampler implementation") # Unskipped
    def test_sample_observational_reproducibility(self):
        """Test observational sampling reproducibility with seed."""
        samples1 = sample_observational(self.scm, num_samples=50, seed=42)
        samples2 = sample_observational(self.scm, num_samples=50, seed=42)
        pd.testing.assert_frame_equal(samples1, samples2)
        # pass # Removed pass

    # @unittest.skip("Needs sampler implementation") # Unskipped
    def test_sample_observational_distribution(self):
        """Test if sampled observational data roughly matches expected distributions."""
        samples = sample_observational(self.scm, num_samples=5000, seed=123)
        # Simple checks based on X ~ N(0,1), Y = 2X + N(0,0.5), Z = -Y + N(0,0.2)
        # Expected means should be close to 0 for X, Y, Z
        self.assertAlmostEqual(samples['X'].mean(), 0, delta=0.15)
        self.assertAlmostEqual(samples['Y'].mean(), 0, delta=0.15)
        self.assertAlmostEqual(samples['Z'].mean(), 0, delta=0.15)
        # Expected stddev
        # Var(Y) = 4*Var(X) + Var(NoiseY) = 4*1 + 0.5^2 = 4.25 => std(Y) ~ 2.06
        # Var(Z) = (-1)^2*Var(Y) + Var(NoiseZ) = Var(Y) + 0.2^2 = 4.25 + 0.04 = 4.29 => std(Z) ~ 2.07
        self.assertAlmostEqual(samples['X'].std(), 1, delta=0.2)
        self.assertAlmostEqual(samples['Y'].std(), np.sqrt(4.25), delta=0.2)
        self.assertAlmostEqual(samples['Z'].std(), np.sqrt(4.29), delta=0.2)
        # pass # Removed pass

    # --- Interventional Sampling Tests ---

    # @unittest.skip("Needs sampler implementation") # Unskipped
    def test_sample_interventional_perfect(self):
        """Test sampling after a perfect intervention."""
        intervention = PerfectIntervention('Y', value=5.0)
        samples = sample_interventional(self.scm, interventions=[intervention], num_samples=100, seed=43)
        self.assertTrue(np.allclose(samples['Y'], 5.0))
        # Check downstream effects on Z = -Y + N(0, 0.2)
        # E[Z | do(Y=5)] = -5
        # Var(Z | do(Y=5)) = Var(NoiseZ) = 0.2^2 = 0.04
        self.assertAlmostEqual(samples['Z'].mean(), -5.0, delta=0.2)
        self.assertAlmostEqual(samples['Z'].std(), 0.2, delta=0.1)
        # X should be unaffected
        self.assertAlmostEqual(samples['X'].mean(), 0, delta=0.2)
        self.assertAlmostEqual(samples['X'].std(), 1, delta=0.25)
        # pass

    # @unittest.skip("Needs sampler implementation") # Unskipped
    def test_sample_interventional_imperfect(self):
        """Test sampling after an imperfect intervention."""
        # Y_orig = 2*X + N_Y
        # Y_int = (1-s)*Y_orig + s*V = (1-0.5)*Y_orig + 0.5*5 = 0.5*(2X+N_Y) + 2.5 = X + 0.5*N_Y + 2.5
        # E[Y_int] = E[X] + 0.5*E[N_Y] + 2.5 = 0 + 0 + 2.5 = 2.5
        # Var(Y_int) = Var(X) + 0.25*Var(N_Y) = 1 + 0.25*0.5^2 = 1 + 0.25*0.25 = 1 + 0.0625 = 1.0625
        intervention = ImperfectIntervention('Y', value=5.0, strength=0.5, combination_method='weighted_average')
        samples = sample_interventional(self.scm, interventions=[intervention], num_samples=2000, seed=44)

        self.assertFalse(np.allclose(samples['Y'], 5.0), "Y should not be fixed at 5.0")
        self.assertFalse(np.allclose(samples['Y'].mean(), 0.0), "Mean of Y should change")
        self.assertAlmostEqual(samples['Y'].mean(), 2.5, delta=0.15)
        self.assertAlmostEqual(samples['Y'].std(), np.sqrt(1.0625), delta=0.15)
        # pass

    # @unittest.skip("Needs sampler implementation") # Unskipped
    def test_sample_interventional_soft(self):
        """Test sampling after a soft intervention (modifying mechanism)."""
        # Original: Y = 2*X + N_Y
        # New: Y = 3*X + N_Y
        def modify_y_mechanism(X, noise, **kwargs): # Added **kwargs to handle potential original_output from compose=True if needed
             return 3 * X + noise
        intervention = SoftIntervention('Y', intervention_function=modify_y_mechanism, compose=False)
        samples_soft = sample_interventional(self.scm, interventions=[intervention], num_samples=2000, seed=45)
        samples_obs = sample_observational(self.scm, num_samples=2000, seed=45) # For comparison

        # Check if Y distribution changed (specifically variance)
        # Var(Y_soft) = 9*Var(X) + Var(N_Y) = 9*1 + 0.5^2 = 9.25
        self.assertGreater(samples_soft['Y'].var(), samples_obs['Y'].var()) # Variance should increase
        self.assertAlmostEqual(samples_soft['Y'].std(), np.sqrt(9.25), delta=0.25)
        # Check effect on Z
        # Var(Z_soft) = Var(Y_soft) + Var(N_Z) = 9.25 + 0.2^2 = 9.25 + 0.04 = 9.29
        self.assertAlmostEqual(samples_soft['Z'].std(), np.sqrt(9.29), delta=0.25)
        # pass

    # @unittest.skip("Needs sampler implementation") # Unskipped
    def test_sample_interventional_multiple(self):
        """Test sampling after multiple simultaneous interventions."""
        interventions = [PerfectIntervention('X', value=1.0), PerfectIntervention('Y', value=-2.0)]
        samples = sample_interventional(self.scm, interventions=interventions, num_samples=100, seed=46)

        self.assertTrue(np.allclose(samples['X'], 1.0))
        self.assertTrue(np.allclose(samples['Y'], -2.0))
        # Check effect on Z = -Y + N(0, 0.2)
        # E[Z | do(X=1, Y=-2)] = -(-2) = 2
        # Var(Z | do(X=1, Y=-2)) = Var(N_Z) = 0.2^2 = 0.04
        self.assertAlmostEqual(samples['Z'].mean(), 2.0, delta=0.15)
        self.assertAlmostEqual(samples['Z'].std(), 0.2, delta=0.1)
        # pass

    # --- Counterfactual Sampling Tests ---

    @unittest.skip("Needs sampler implementation")
    def test_sample_counterfactual_simple(self):
        """Test a simple counterfactual query."""
        # factual_evidence = {'X': 0.5} # Example observation
        # counterfactual_intervention = {'X': -1.0} # What if X had been -1.0?
        # cf_samples = sample_counterfactual(self.scm,
        #                                     factual_evidence=factual_evidence,
        #                                     counterfactual_interventions=counterfactual_intervention,
        #                                     num_samples=100)
        # Check distribution of Y and Z under the counterfactual
        pass

    # --- Batch Sampling Tests ---

    @unittest.skip("Needs sampler implementation")
    def test_batch_observational_sampling(self):
        """Test efficient batch observational sampling."""
        # samples = sample_observational(self.scm, num_samples=500, batch_size=100)
        # self.assertEqual(samples.shape[0], 500)
        pass

    @unittest.skip("Needs sampler implementation")
    def test_batch_interventional_sampling(self):
        """Test efficient batch interventional sampling."""
        # intervention = PerfectIntervention('Y', value=5.0)
        # samples = sample_interventional(self.scm, interventions=[intervention], num_samples=500, batch_size=100)
        # self.assertEqual(samples.shape[0], 500)
        # self.assertTrue(np.allclose(samples['Y'], 5.0))
        pass


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False) 