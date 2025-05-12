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
    scm.add_variable("x0")
    scm.add_variable("x1")
    scm.add_variable("x2")
    scm.graph.add_edge("x0", "x1")
    scm.graph.add_edge("x1", "x2")

    # Define linear mechanisms
    # Exogenous
    scm.structural_equations["x0"] = lambda **parents: np.random.normal(0, 1)
    scm.structural_equations["x1"] = lambda x0: 2 * x0 + np.random.normal(0, 0.1)
    scm.structural_equations["x2"] = lambda x1: -1 * x1 + np.random.normal(0, 0.1)

    return scm

# --- Tests for PerfectIntervention ---


def test_perfect_intervention_apply():
    """Test applying a perfect intervention (do-operator)."""
    scm_orig = create_simple_scm()
    target_node = "x1"
    intervention_value = 10.0

    intervention = PerfectIntervention(
        target_node=target_node, value=intervention_value)
    scm_intervened = intervention.apply(scm_orig)

    # 1. Verify it's a new object (deep copy)
    assert scm_intervened is not scm_orig
    assert scm_intervened.graph is not scm_orig.graph
    assert scm_intervened.structural_equations is not scm_orig.structural_equations

    # 2. Verify graph modification: parent edge x0->x1 removed
    assert not scm_intervened.graph.has_edge(
        "x0", "x1"), "Parent edge x0->x1 should be removed."
    assert scm_intervened.graph.has_edge(
        "x1", "x2"), "Edge x1->x2 should still exist."
    assert set(scm_intervened.graph.get_parents("x1")) == set(
    ), "Node x1 should have no parents after intervention."

    # 3. Verify structural equation modification: x1 is now constant
    # Sample inputs (don't matter for constant function)
    assert scm_intervened.structural_equations[target_node](
        x0=5) == intervention_value
    assert scm_intervened.structural_equations[target_node](
        x0=-2) == intervention_value

    # 4. Verify other equations are unchanged
    assert scm_intervened.structural_equations["x0"] is scm_orig.structural_equations["x0"]
    # Note: deepcopy might affect this check depending on how functions are copied
    assert scm_intervened.structural_equations["x2"] is scm_orig.structural_equations["x2"]

    # 5. Verify original SCM is unchanged
    assert scm_orig.graph.has_edge("x0", "x1")
    assert set(scm_orig.graph.get_parents("x1")) == {"x0"}
    # Check original mechanism (might need a more robust check depending on deepcopy)
    # For simplicity, we rely on the fact that the dictionary was copied.


def test_perfect_intervention_invalid_node():
    """Test applying a perfect intervention to a non-existent node."""
    scm = create_simple_scm()
    bad_intervention = PerfectIntervention(target_node="x3", value=5.0)
    with pytest.raises(ValueError, match="Target node 'x3' not found"):
        bad_intervention.apply(scm)


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
            MockIntervention(target_node="x1")


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
    target_node = "x1"
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
    assert scm_intervened.graph.has_edge("x0", "x1")
    assert set(scm_intervened.graph.get_parents("x1")) == {"x0"}

    # 3. Verify structural equation modification
    # We check the approximate output, ignoring the small random noise from original mechanism
    # Note: The original mechanism is 2*x0 + noise(0, 0.1)
    calculated_y = scm_intervened.structural_equations[target_node](x0=x_val)
    # Use approx for comparison due to noise in the original mechanism part
    assert calculated_y == pytest.approx(
        expected_y_approx, abs=0.5)  # Allow tolerance for noise

    # 4. Verify other equations are unchanged
    assert scm_intervened.structural_equations["x0"] is scm_orig.structural_equations["x0"]
    # x2 depends on x1, so its output will change, but the function itself should be the same reference (if deepcopy allows)
    assert scm_intervened.structural_equations["x2"] is scm_orig.structural_equations["x2"]


def test_imperfect_intervention_non_numeric_fallback():
    """Test fallback behavior for non-numeric values."""
    scm = StructuralCausalModel()
    scm.add_variable("x0")
    scm.add_variable("x1")
    scm.add_variable("x2")
    scm.graph.add_edge("x0", "x1")
    scm.structural_equations["x0"] = lambda **p: 'cat'
    scm.structural_equations["x1"] = lambda x0: x0 + "_orig"

    intervention_value = 'dog'

    # Weighted average falls back to intervention if strength >= 0.5 (approx)
    intv1 = ImperfectIntervention(
        "x1", intervention_value, strength=0.8, combination_method='weighted_average')
    scm1 = intv1.apply(scm)
    assert scm1.structural_equations["x1"](x0='cat') == 'dog'

    intv2 = ImperfectIntervention(
        "x1", intervention_value, strength=0.3, combination_method='weighted_average')
    scm2 = intv2.apply(scm)
    assert scm2.structural_equations["x1"](x0='cat') == 'cat_orig'

    # Additive/Multiplicative fall back based on strength threshold 0.5
    intv3 = ImperfectIntervention(
        "x1", intervention_value, strength=0.6, combination_method='additive')
    scm3 = intv3.apply(scm)
    assert scm3.structural_equations["x1"](x0='cat') == 'dog'

    intv4 = ImperfectIntervention(
        "x1", intervention_value, strength=0.4, combination_method='multiplicative')
    scm4 = intv4.apply(scm)
    assert scm4.structural_equations["x1"](x0='cat') == 'cat_orig'


def test_imperfect_intervention_init_validation():
    """Test initialization validation for ImperfectIntervention."""
    # Missing value
    with pytest.raises(ValueError, match="requires a 'value' keyword"):
        ImperfectIntervention(target_node="x1", strength=0.5)
    # Invalid strength
    with pytest.raises(ValueError, match="strength must be a float between 0.0 and 1.0"):
        ImperfectIntervention(target_node="x1", value=1, strength=-0.1)
    with pytest.raises(ValueError, match="strength must be a float between 0.0 and 1.0"):
        ImperfectIntervention(target_node="x1", value=1, strength=1.1)
    # Invalid combination method
    with pytest.raises(ValueError, match="combination_method must be one of"):
        ImperfectIntervention(target_node="x1", value=1,
                              combination_method="average")


# --- Tests for SoftIntervention ---

def test_soft_intervention_apply_replace():
    """Test applying a soft intervention that replaces the mechanism."""
    scm_orig = create_simple_scm()
    target_node = "x1"

    def new_x1_mechanism(x0):  # Note: accepts parent name explicitly
        return 100 + x0 * 0.1  # Completely different logic

    intervention = SoftIntervention(
        target_node=target_node, intervention_function=new_x1_mechanism, compose=False)
    scm_intervened = intervention.apply(scm_orig)

    # 1. Verify graph unchanged
    assert scm_intervened.graph.has_edge("x0", "x1")
    assert set(scm_intervened.graph.get_parents("x1")) == {"x0"}

    # 2. Verify mechanism is replaced
    assert scm_intervened.structural_equations[target_node](
        x0=10) == 100 + 10 * 0.1
    assert scm_intervened.structural_equations[target_node](
        x0=0) == 100 + 0 * 0.1
    assert scm_intervened.structural_equations[target_node] is not scm_orig.structural_equations[target_node]

    # 3. Verify other equations unchanged
    assert scm_intervened.structural_equations["x0"] is scm_orig.structural_equations["x0"]
    assert scm_intervened.structural_equations["x2"] is scm_orig.structural_equations["x2"]


def test_soft_intervention_apply_compose():
    """Test applying a soft intervention that composes with the original mechanism."""
    scm_orig = create_simple_scm()
    target_node = "x1"

    # This function will receive the original output and parent values
    def compose_func(original_output, x0, scale_factor):  # Also takes func_kwargs
        # E.g., scale the original output based on parent x0 and a fixed factor
        return original_output * scale_factor + x0

    intervention = SoftIntervention(
        target_node=target_node,
        intervention_function=compose_func,
        compose=True,
        scale_factor=0.5  # Pass extra kwarg
    )
    scm_intervened = intervention.apply(scm_orig)

    # 1. Verify graph unchanged
    assert scm_intervened.graph.has_edge("x0", "x1")

    # 2. Verify mechanism is composed
    # Original x1 approx = 2*x0. Composed x1 approx = (2*x0) * 0.5 + x0 = x0 + x0 = 2*x0
    # Let's try different values
    x_val1 = 1.0
    original_x1_1 = scm_orig.structural_equations[target_node](
        x0=x_val1)  # Approx 2.0
    expected_x1_1 = original_x1_1 * 0.5 + x_val1
    calculated_x1_1 = scm_intervened.structural_equations[target_node](x0=x_val1)
    assert calculated_x1_1 == pytest.approx(expected_x1_1, abs=0.1)

    x_val2 = 5.0
    original_x1_2 = scm_orig.structural_equations[target_node](
        x0=x_val2)  # Approx 10.0
    expected_x1_2 = original_x1_2 * 0.5 + x_val2
    calculated_x1_2 = scm_intervened.structural_equations[target_node](x0=x_val2)
    assert calculated_x1_2 == pytest.approx(expected_x1_2, abs=0.1)


def test_soft_intervention_init_validation():
    """Test initialization validation for SoftIntervention."""
    # Missing function
    with pytest.raises(ValueError, match="requires a callable 'intervention_function'"):
        SoftIntervention(target_node="x1", intervention_function=None)
    # Non-callable function
    with pytest.raises(ValueError, match="intervention_function must be a callable"):
        SoftIntervention(target_node="x1", intervention_function=5)
    # Invalid compose type
    with pytest.raises(ValueError, match="'compose' must be a boolean"):
        SoftIntervention(
            target_node="x1", intervention_function=lambda x: x, compose="True")


def test_soft_intervention_apply_compose_missing_orig():
    """Test compose=True when original mechanism doesn't exist."""
    scm = StructuralCausalModel()
    scm.add_variable("x0")
    scm.add_variable("x1")
    scm.add_variable("x2")
    scm.graph.add_edge("x0", "x1")
    # No structural equation defined for x1

    intervention = SoftIntervention("x1", lambda x0: x0+1, compose=True)
    with pytest.raises(ValueError, match="Original structural equation for 'x1' not found"):
        intervention.apply(scm)
