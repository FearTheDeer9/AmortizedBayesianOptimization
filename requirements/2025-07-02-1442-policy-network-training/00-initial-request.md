# Initial Request

**Date**: 2025-07-02  
**Time**: 14:42  
**Slug**: policy-network-training

## User Request

We are now at the point where we are ready to begin training our surrogate model and our policy model for the first time using behavioural cloning and GRPO as well as potential joint training afterwards, as specified in docs/training. Lets start with our policy network training- look at our models at src/causal_bayes_opt/acquisition/enhanced_policy_network.py as well as the initial training script at scripts/run_3min_validation.py scripts/train_acquisition_grpo.py scripts/train_acquisition_surrogate_free.py and figure out how we can ensure that the training is properly set up, i.e that there are no bugs in the loop and that the training system is giving correct signals so as to promote actual improvement at the relevant tasks (selecting interventions that promote structure learning as well as target value optimization) - once we do this we can run a more comprehensive training run and optimize our model

## Context from Initial Analysis

- Project has sophisticated GRPO training infrastructure
- Enhanced policy networks with attention mechanisms exist
- Multiple training scripts available
- Training pipeline documented in docs/training/
- Goal is to validate training setup before full training runs
- Focus on policy network training specifically
- Need to ensure learning signals promote actual improvement