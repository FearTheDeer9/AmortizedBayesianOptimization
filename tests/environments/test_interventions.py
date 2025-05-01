import pytest
import numpy as np
import copy

from causal_meta.environments.scm import StructuralCausalModel
from causal_meta.environments.interventions import PerfectIntervention, ImperfectIntervention, SoftIntervention
from causal_meta.graph import DirectedGraph

# Helper function to create a simple SCM for testing


def create_simple_scm():
    """Creates a simple linear SCM: X -> Y -> Z"""
    scm = StructuralCausalModel()
    scm.add_variable("X")
    scm.add_variable("Y")
    scm.add_variable("Z")
    scm.graph.add_edge("X", "Y")
    scm.graph.add_edge("Y", "Z")

    # Define linear mechanisms
    # Exogenous
    scm.structural_equations["X"] = lambda **parents: np.random.normal(0, 1)
    scm.structural_equations["Y"] = lambda X: 2 * X + np.random.normal(0, 0.1)
    scm.structural_equations["Z"] = lambda Y: -1 * Y + np.random.normal(0, 0.1)

    return scm

# --- Tests for PerfectIntervention ---


def test_perfect_intervention_apply():
    """Test applying a perfect intervention (do-operator)."""
    scm_orig = create_simple_scm()
    target_node = "Y"
    intervention_value = 10.0

    intervention = PerfectIntervention(
        target_node=target_node, value=intervention_value)
    scm_intervened = intervention.apply(scm_orig)

    # 1. Verify it's a new object (deep copy)
    assert scm_intervened is not scm_orig
    assert scm_intervened.graph is not scm_orig.graph
    assert scm_intervened.structural_equations is not scm_orig.structural_equations

    # 2. Verify graph modification: parent edge X->Y removed
    assert not scm_intervened.graph.has_edge(
        "X", "Y"), "Parent edge X->Y should be removed."
    assert scm_intervened.graph.has_edge(
        "Y", "Z"), "Edge Y->Z should still exist."
    assert set(scm_intervened.graph.get_parents("Y")) == set(
    ), "Node Y should have no parents after intervention."

    # 3. Verify structural equation modification: Y is now constant
    # Sample inputs (don't matter for constant function)
    assert scm_intervened.structural_equations[target_node](
        X=5) == intervention_value
    assert scm_intervened.structural_equations[target_node](
        X=-2) == intervention_value

    # 4. Verify other equations are unchanged
    assert scm_intervened.structural_equations["X"] is scm_orig.structural_equations["X"]
    # Note: deepcopy might affect this check depending on how functions are copied
    assert scm_intervened.structural_equations["Z"] is scm_orig.structural_equations["Z"]

    # 5. Verify original SCM is unchanged
    assert scm_orig.graph.has_edge("X", "Y")
    assert set(scm_orig.graph.get_parents("Y")) == {"X"}
    # Check original mechanism (might need a more robust check depending on deepcopy)
    # For simplicity, we rely on the fact that the dictionary was copied.


def test_perfect_intervention_invalid_node():
    """Test applying a perfect intervention to a non-existent node."""
    scm = create_simple_scm()
    intervention = PerfectIntervention(target_node="W", value=5.0)
    with pytest.raises(ValueError, match="Target node 'W' not found"):
        intervention.apply(scm)


def test_perfect_intervention_init_validation():
    """Test initialization validation for PerfectIntervention."""
    with pytest.raises(ValueError, match="target_node must be a non-empty string"):
        PerfectIntervention(target_node="", value=1)
    with pytest.raises(ValueError, match="requires a 'value' keyword argument"):
        # Need to bypass standard init to test _validate_kwargs directly if possible,
        # or trigger it indirectly. This checks if the mechanism validation works.
        # Let's test via the base class init call path if it catches it.
        class MockIntervention(PerfectIntervention):
            def __init__(self, target_node):
                # Call grandparent init to bypass PerfectIntervention's specific check
                # then explicitly call validation. This is a bit hacky for testing.
                Intervention.__init__(
                    self, target_node=target_node)  # Call base init
                self._validate_kwargs()  # Manually call validation without value

        with pytest.raises(ValueError, match="requires a 'value' keyword argument"):
            MockIntervention(target_node="Y")


# --- Tests for ImperfectIntervention ---

@pytest.mark.parametrize(
    "combination_method, strength, x_val, expected_y_approx", [
        ("additive", 1.0, 1.0, 2.0 + 1.0 * 5.0),  # 2*1 + 1.0*5 = 7
        ("additive", 0.5, 2.0, 4.0 + 0.5 * 5.0),  # 2*2 + 0.5*5 = 6.5
        ("multiplicative", 1.0, 1.0, 2.0 *
         (1 + 1.0 * (5.0 - 1))),  # 2 * (1 + 1*4) = 10
        ("multiplicative", 0.5, 2.0, 4.0 *
         (1 + 0.5 * (5.0 - 1))),  # 4 * (1 + 0.5*4) = 12
        ("weighted_average", 1.0, 1.0, 5.0),  # 0*2 + 1*5 = 5
        ("weighted_average", 0.75, 2.0, (1-0.75)*4.0 +
         0.75*5.0),  # 0.25*4 + 0.75*5 = 1 + 3.75 = 4.75
        ("weighted_average", 0.0, 1.0, 2.0),  # 1*2 + 0*5 = 2
    ]
)
def test_imperfect_intervention_apply(combination_method, strength, x_val, expected_y_approx):
    """Test applying imperfect interventions with different methods and strengths."""
    scm_orig = create_simple_scm()
    target_node = "Y"
    intervention_value = 5.0

    intervention = ImperfectIntervention(
        target_node=target_node,
        value=intervention_value,
        strength=strength,
        combination_method=combination_method
    )
    scm_intervened = intervention.apply(scm_orig)

    # 1. Verify it's a new object
    assert scm_intervened is not scm_orig

    # 2. Verify graph is UNCHANGED (parents are kept)
    assert scm_intervened.graph.has_edge("X", "Y")
    assert set(scm_intervened.graph.get_parents("Y")) == {"X"}

    # 3. Verify structural equation modification
    # We check the approximate output, ignoring the small random noise from original mechanism
    # Note: The original mechanism is 2*X + noise(0, 0.1)
    calculated_y = scm_intervened.structural_equations[target_node](X=x_val)
    # Use approx for comparison due to noise in the original mechanism part
    assert calculated_y == pytest.approx(
        expected_y_approx, abs=0.5)  # Allow tolerance for noise

    # 4. Verify other equations are unchanged
    assert scm_intervened.structural_equations["X"] is scm_orig.structural_equations["X"]
    # Z depends on Y, so its output will change, but the function itself should be the same reference (if deepcopy allows)
    assert scm_intervened.structural_equations["Z"] is scm_orig.structural_equations["Z"]


def test_imperfect_intervention_non_numeric_fallback():
    """Test fallback behavior for non-numeric values."""
    scm = StructuralCausalModel()
    scm.add_variable("A")
    scm.add_variable("B")
    scm.graph.add_edge("A", "B")
    scm.structural_equations["A"] = lambda **p: 'cat'
    scm.structural_equations["B"] = lambda A: A + "_orig"

    intervention_value = 'dog'

    # Weighted average falls back to intervention if strength >= 0.5 (approx)
    intv1 = ImperfectIntervention(
        "B", intervention_value, strength=0.8, combination_method='weighted_average')
    scm1 = intv1.apply(scm)
    assert scm1.structural_equations["B"](A='cat') == 'dog'

    intv2 = ImperfectIntervention(
        "B", intervention_value, strength=0.3, combination_method='weighted_average')
    scm2 = intv2.apply(scm)
    assert scm2.structural_equations["B"](A='cat') == 'cat_orig'

    # Additive/Multiplicative fall back based on strength threshold 0.5
    intv3 = ImperfectIntervention(
        "B", intervention_value, strength=0.6, combination_method='additive')
    scm3 = intv3.apply(scm)
    assert scm3.structural_equations["B"](A='cat') == 'dog'

    intv4 = ImperfectIntervention(
        "B", intervention_value, strength=0.4, combination_method='multiplicative')
    scm4 = intv4.apply(scm)
    assert scm4.structural_equations["B"](A='cat') == 'cat_orig'


def test_imperfect_intervention_init_validation():
    """Test initialization validation for ImperfectIntervention."""
    # Missing value
    with pytest.raises(ValueError, match="requires a 'value' keyword"):
        ImperfectIntervention(target_node="Y", strength=0.5)
    # Invalid strength
    with pytest.raises(ValueError, match="strength must be a float between 0.0 and 1.0"):
        ImperfectIntervention(target_node="Y", value=1, strength=-0.1)
    with pytest.raises(ValueError, match="strength must be a float between 0.0 and 1.0"):
        ImperfectIntervention(target_node="Y", value=1, strength=1.1)
    # Invalid combination method
    with pytest.raises(ValueError, match="combination_method must be one of"):
        ImperfectIntervention(target_node="Y", value=1,
                              combination_method="average")


# --- Tests for SoftIntervention ---

def test_soft_intervention_apply_replace():
    """Test applying a soft intervention that replaces the mechanism."""
    scm_orig = create_simple_scm()
    target_node = "Y"

    def new_y_mechanism(X):  # Note: accepts parent name explicitly
        return 100 + X * 0.1  # Completely different logic

    intervention = SoftIntervention(
        target_node=target_node, intervention_function=new_y_mechanism, compose=False)
    scm_intervened = intervention.apply(scm_orig)

    # 1. Verify graph unchanged
    assert scm_intervened.graph.has_edge("X", "Y")
    assert set(scm_intervened.graph.get_parents("Y")) == {"X"}

    # 2. Verify mechanism is replaced
    assert scm_intervened.structural_equations[target_node](
        X=10) == 100 + 10 * 0.1
    assert scm_intervened.structural_equations[target_node](
        X=0) == 100 + 0 * 0.1
    assert scm_intervened.structural_equations[target_node] is not scm_orig.structural_equations[target_node]

    # 3. Verify other equations unchanged
    assert scm_intervened.structural_equations["X"] is scm_orig.structural_equations["X"]
    assert scm_intervened.structural_equations["Z"] is scm_orig.structural_equations["Z"]


def test_soft_intervention_apply_compose():
    """Test applying a soft intervention that composes with the original mechanism."""
    scm_orig = create_simple_scm()
    target_node = "Y"

    # This function will receive the original output and parent values
    def compose_func(original_output, X, scale_factor):  # Also takes func_kwargs
        # E.g., scale the original output based on parent X and a fixed factor
        return original_output * scale_factor + X

    intervention = SoftIntervention(
        target_node=target_node,
        intervention_function=compose_func,
        compose=True,
        scale_factor=0.5  # Pass extra kwarg
    )
    scm_intervened = intervention.apply(scm_orig)

    # 1. Verify graph unchanged
    assert scm_intervened.graph.has_edge("X", "Y")

    # 2. Verify mechanism is composed
    # Original Y approx = 2*X. Composed Y approx = (2*X) * 0.5 + X = X + X = 2*X
    # Let's try different values
    x_val1 = 1.0
    original_y1 = scm_orig.structural_equations[target_node](
        X=x_val1)  # Approx 2.0
    expected_y1 = original_y1 * 0.5 + x_val1
    calculated_y1 = scm_intervened.structural_equations[target_node](X=x_val1)
    assert calculated_y1 == pytest.approx(expected_y1, abs=0.1)

    x_val2 = 5.0
    original_y2 = scm_orig.structural_equations[target_node](
        X=x_val2)  # Approx 10.0
    expected_y2 = original_y2 * 0.5 + x_val2
    calculated_y2 = scm_intervened.structural_equations[target_node](X=x_val2)
    assert calculated_y2 == pytest.approx(expected_y2, abs=0.1)


def test_soft_intervention_init_validation():
    """Test initialization validation for SoftIntervention."""
    # Missing function
    with pytest.raises(ValueError, match="requires a callable 'intervention_function'"):
        SoftIntervention(target_node="Y", intervention_function=None)
    # Non-callable function
    with pytest.raises(ValueError, match="intervention_function must be a callable"):
        SoftIntervention(target_node="Y", intervention_function=5)
    # Invalid compose type
    with pytest.raises(ValueError, match="'compose' must be a boolean"):
        SoftIntervention(
            target_node="Y", intervention_function=lambda x: x, compose="True")


def test_soft_intervention_apply_compose_missing_orig():
    """Test compose=True when original mechanism doesn't exist."""
    scm = StructuralCausalModel()
    scm.add_variable("A")
    scm.add_variable("B")
    scm.graph.add_edge("A", "B")
    # No structural equation defined for B

    intervention = SoftIntervention("B", lambda A: A+1, compose=True)
    with pytest.raises(ValueError, match="Original structural equation for 'B' not found"):
        intervention.apply(scm)
