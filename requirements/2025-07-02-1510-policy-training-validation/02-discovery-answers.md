# Discovery Answers

## Q1: Should the validation comprehensively test all the major changes we made (reward system, silent failure fixes, state tensor creation, component validation) rather than focusing primarily on one area?
**Answer:** Yes - comprehensive validation ensures all components work together properly

## Q2: Do you want to run adversarial tests to specifically try to "hack" or exploit the new reward system?
**Answer:** Yes - important to identify potential gaming strategies before full training

## Q3: Should the validation include comparing the new reward system against the old relative-improvement reward on sample scenarios?
**Answer:** No - delete the old approach to avoid dead code and confusion about methodology. Document this decision in docs/training/

## Q4: Do you want to test the reward normalization scheme across different variable value ranges and intervention bounds?
**Answer:** Yes - test different variable ranges (only linear mechanisms available currently)

## Q5: Should we run actual short training trials to validate that policies improve with the new system before setting up full training configuration?
**Answer:** Yes - empirical validation that learning actually occurs with new rewards