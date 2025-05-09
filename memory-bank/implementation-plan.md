# Implementation Plan

This document reflects the updated plan for the project based on our immediate goal: creating an MVP demonstration for neural network-based causal structure learning with meta-learning.

## UPDATED PRIORITY: MVP Neural Network Demo

Our immediate priority is to develop an MVP demonstration that shows how neural networks with meta-learning can successfully recover the true causal graph structure using both observational and interventional data. Only after this key milestone is achieved will we return to the broader refactoring tasks.

## MVP Demo Tasks (Priority)

### Task A: Neural Causal Discovery MVP (High Priority)

- **Description:** Create a working demo that shows neural networks can recover true graph structure through meta-learning and adaptive interventions
- **Priority:** HIGHEST
- **Status:** in-progress
- **Dependencies:** none

- **Subtask A.1: Improve Progressive Structure Recovery Demo (in-progress)**
  - **Sequential Thinking Analysis:**
    - **Thought 1: Problem Understanding**
      - Need to fix issues in current progressive_structure_recovery_demo.py
      - Must properly encode intervention information for the model
      - Need to evaluate and improve convergence properties for larger graphs
      - Must make the demo more robust for graphs with more than 3 nodes

    - **Thought 2: Component Identification**
      - Current AmortizedCausalDiscovery model
      - MAML adaptation mechanism
      - Intervention representation
      - Graph comparison utilities

    - **Thought 3: Implementation Approach**
      - Modify the model inputs to explicitly encode intervention information
      - Improve uncertainty estimates for better intervention selection
      - Add regularization to prevent overfitting during adaptation
      - Add clear visualization of learning progress

    - **Thought 4: Potential Challenges**
      - Encoding interventions appropriately for the neural network
      - Balancing exploration and exploitation in intervention selection
      - Ensuring the model adapts correctly to intervention data
      - Preventing catastrophic forgetting during adaptation

    - **Thought 5: Implementation Plan**
      - Update intervention encoding in model inputs to clearly mark interventions
      - Implement better uncertainty-based intervention selection
      - Add regularization during adaptation to prevent overfitting
      - Improve visualization of learning progress
      - Test with different graph sizes and complexities

  - **Detailed Implementation Steps:**
    1. Modify the progressive_structure_recovery_demo.py to explicitly encode intervention information
    2. Implement an improved intervention selection strategy using uncertainty estimates
    3. Add regularization to the MAML adaptation process
    4. Enhance visualization of learning progress (SHD history, graph evolution)
    5. Add comprehensive testing across different graph sizes
    6. Document the approach and results

  - **Implementation Progress:**
    1. Created a new `EnhancedMAMLForCausalDiscovery` class that properly encodes intervention information during adaptation
    2. Updated `format_interventions` function to create binary intervention masks for the model
    3. Enhanced `select_intervention_node` function to use edge uncertainty for better intervention selection
    4. Updated `infer_adjacency_matrix` to handle intervention information 
    5. Added L2 and sparsity regularization during model adaptation to prevent overfitting
    6. Enhanced visualization with uncertainty plots and detailed tracking of learning progress
    7. Added better handling of edge uncertainties to focus interventions on the most uncertain parts of the graph
    8. Added command-line parameters to control regularization and other adaptation parameters
    9. Created a simplified causal discovery model (`SimplifiedCausalDiscovery`) that resolves tensor compatibility issues
    10. Fixed node ID handling in intervention selection to prevent errors with different graph sizes
    11. Implemented a testing framework for multiple graph sizes to evaluate scaling properties
    12. Successfully demonstrated convergence to true graph structure on small graphs (3 nodes)
    13. Verified the approach works on graphs of different sizes (3-5 nodes tested)
    14. Enhanced the SimplifiedCausalDiscovery model with attention mechanisms for better performance on larger graphs
    15. Added annealing to regularization parameters during adaptation to balance exploration and exploitation
    16. Implemented acyclicity constraints to enforce valid DAG structures
    17. Created a comprehensive evaluation framework (evaluate_structure_recovery.py) to test different configurations
    18. Added multi-head attention mechanism that scales with graph size for better node interactions
    19. Improved the uncertainty-based intervention selection with combined strategies 
    20. Added visualization of intervention patterns across different graph sizes
    21. Created detailed tracking of edge uncertainties during the learning process 
    22. Enhanced the command line interface to support systematic experimentation

  - **Implementation Challenges:**
    1. Encountered compatibility issues with the AmortizedCausalDiscovery class interface
    2. The model requires complex inputs (node_features, edge_index, batch) that need to be properly constructed
    3. Matrix shape mismatches between model expectations and input data
    4. Need to adapt the structural causal model implementation to match available components
    5. Node ID format inconsistencies causing errors in the intervention selection process
    6. Uncertainty estimation varies in quality across different graph sizes
    7. Model performance degradation as graph size increases
    8. Attention mechanism adds computational complexity that scales with graph size
    9. Balancing regularization strength to avoid underfitting/overfitting during adaptation
    10. Finding the right uncertainty threshold for intervention selection across different graph sizes
    11. Preventing the model from getting stuck in local optima during adaptation

  - **Next Steps:**
    1. Run comprehensive evaluations to benchmark performance across different graph sizes
    2. Fine-tune regularization parameters based on evaluation results
    3. Explore more advanced attention mechanisms for even larger graphs (>10 nodes)
    4. Implement additional regularization approaches to stabilize learning on larger graphs
    5. Test with more complex graph structures and a wider range of SCM parameters
    6. Add adaptive learning rate schedule for more stable adaptation
    7. Explore pruning strategies to eliminate weak edges during adaptation

  - **Current Status:** in-progress
  - **Estimated Completion:** 1 week

- **Subtask A.2: Optimize Meta-Learning Parameters (pending)**
  - **Sequential Thinking Analysis:**
    - **Thought 1: Problem Understanding**
      - Need to find optimal meta-learning parameters for convergence
      - Different graph sizes and complexities may require different parameters
      - Must balance adaptation speed and stability
      - Need to ensure the model doesn't overfit to specific interventions

    - **Thought 2: Component Identification**
      - MAML inner learning rate
      - Number of inner adaptation steps
      - Regularization parameters
      - Model architecture parameters

    - **Thought 3: Implementation Approach**
      - Conduct grid search or Bayesian optimization for key parameters
      - Test different combinations of parameters on various graph sizes
      - Monitor convergence rate and final accuracy
      - Identify robust parameter settings that work across graph types

    - **Thought 4: Potential Challenges**
      - Parameter search is computationally expensive
      - Optimal parameters may vary significantly across graph types
      - Need to avoid local optima in parameter space
      - Must balance exploration and exploitation in parameter search

    - **Thought 5: Implementation Plan**
      - Define parameter search space (learning rates, steps, regularization)
      - Implement grid search or Bayesian optimization routine
      - Test parameters on small, medium, and large graphs
      - Analyze results and identify robust parameter settings
      - Document findings and optimal configurations

  - **Detailed Implementation Steps:**
    1. Define parameter search space for meta-learning hyperparameters
    2. Implement parameter optimization routine (grid search or Bayesian optimization)
    3. Run tests on different graph sizes and structures
    4. Analyze convergence and accuracy metrics
    5. Document optimal parameter settings for different scenarios
    6. Update the demo with best-performing parameters

  - **Current Status:** pending
  - **Estimated Completion:** 1 week

- **Subtask A.3: Enhance Intervention Strategy (pending)**
  - **Sequential Thinking Analysis:**
    - **Thought 1: Problem Understanding**
      - Current intervention strategies may not be optimal for structure discovery
      - Need to develop strategies that maximize information gain
      - Different strategies may be optimal at different stages of learning
      - Must balance exploration (uncertain edges) and exploitation (confirming hypotheses)

    - **Thought 2: Component Identification**
      - Current intervention selection strategies
      - Uncertainty estimation for edges
      - Information gain calculations
      - Adaptive strategy selection

    - **Thought 3: Implementation Approach**
      - Implement information-theoretic intervention selection
      - Add adaptive strategies that change based on learning progress
      - Incorporate model uncertainty in intervention decisions
      - Compare performance across different strategies

    - **Thought 4: Potential Challenges**
      - Computing optimal interventions may be computationally expensive
      - Balancing exploration and exploitation is non-trivial
      - Strategy effectiveness may vary by graph type and size
      - Need to avoid getting stuck in local optima

    - **Thought 5: Implementation Plan**
      - Implement several intervention strategies (information gain, uncertainty-based, etc.)
      - Add adaptive strategy switching based on learning progress
      - Test strategies on various graph types and sizes
      - Compare convergence rates and final accuracy
      - Document performance characteristics of different strategies

  - **Detailed Implementation Steps:**
    1. Implement information-theoretic intervention selection
    2. Develop uncertainty-weighted intervention strategies
    3. Create adaptive strategy selection mechanism
    4. Benchmark different strategies on various graph types
    5. Analyze convergence rates and final accuracy
    6. Integrate best-performing strategies into the demo
    7. Document findings and recommendations

  - **Current Status:** pending
  - **Estimated Completion:** 1-2 weeks

### Task B: Neural Network Model Improvements (Medium Priority)

- **Description:** Enhance the neural network models to better handle causal discovery tasks
- **Priority:** medium
- **Status:** pending
- **Dependencies:** Task A (partial)

- **Subtask B.1: Improve Intervention Encoding (pending)**
  - **Detailed Implementation Steps:**
    1. Modify the neural network architecture to explicitly handle intervention information
    2. Add specific encoding for interventional vs. observational data
    3. Test different encoding mechanisms and compare performance
    4. Document the most effective encoding approach

  - **Current Status:** pending
  - **Estimated Completion:** 1 week

- **Subtask B.2: Enhance Uncertainty Quantification (pending)**
  - **Detailed Implementation Steps:**
    1. Implement multiple uncertainty estimation methods (ensemble, dropout, etc.)
    2. Add calibration for uncertainty estimates
    3. Test uncertainty quality on various graph types
    4. Integrate best methods into the demo

  - **Current Status:** pending
  - **Estimated Completion:** 1 week

## Original Refactoring Tasks (Lower Priority)

The following tasks from our original implementation plan are now lower priority and will be addressed after the MVP demo is completed successfully:

### Task 1: Core Interface Design (Lower Priority)
- **Status:** partially completed, paused
- **Resume after:** Task A and B are completed

### Task 2: CausalGraph & DirectedGraph Refactoring (Lower Priority)
- **Status:** pending
- **Resume after:** Task A and B are completed

### Task 3: StructuralCausalModel Refactoring (Lower Priority)
- **Status:** pending
- **Resume after:** Task A and B are completed

### Task 4: Structure Inference Models Refactoring (Lower Priority)
- **Status:** partially completed, paused
- **Resume after:** Task A and B are completed

### Task 5: Dynamics Prediction Models Refactoring (Lower Priority)
- **Status:** partially completed, paused
- **Resume after:** Task A and B are completed

### Task 6: Acquisition Strategies Refactoring (Lower Priority)
- **Status:** pending
- **Resume after:** Task A and B are completed

### Remaining Tasks (7-15) (Lower Priority)
- **Status:** pending
- **Resume after:** Task A and B are completed

## Updated Timeline

1. **Task A: Neural Causal Discovery MVP** - 2-3 weeks
   - Subtask A.1: Improve Progressive Structure Recovery Demo - 1 week
   - Subtask A.2: Optimize Meta-Learning Parameters - 1 week
   - Subtask A.3: Enhance Intervention Strategy - 1-2 weeks

2. **Task B: Neural Network Model Improvements** - 2 weeks
   - Subtask B.1: Improve Intervention Encoding - 1 week
   - Subtask B.2: Enhance Uncertainty Quantification - 1 week

3. **Return to Original Refactoring Tasks** - After MVP completion

## Success Criteria for MVP Demo

The MVP demo will be considered successful when:

1. It can consistently recover the true causal structure of:
   - Small graphs (3-4 nodes) with 100% accuracy
   - Medium graphs (5-7 nodes) with at least 90% accuracy
   - Larger graphs (8+ nodes) with reasonable accuracy (>80%)

2. It demonstrates clear learning progress:
   - Structural Hamming Distance decreases consistently with interventions
   - Final graph structure closely matches the true structure
   - Learning is stable and reproducible across different random seeds

3. It provides clear visualizations and metrics:
   - Visual comparison of true vs. inferred graphs
   - SHD history showing learning progress
   - Intervention selection strategy effectiveness

This updated implementation plan prioritizes the creation of a functional MVP demo that demonstrates our neural network approach to causal discovery before returning to the broader refactoring tasks.