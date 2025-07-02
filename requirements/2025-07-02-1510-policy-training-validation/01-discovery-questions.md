# Discovery Questions

## Q1: Should the validation focus primarily on testing the new continuous reward system's behavior under different scenarios?
**Default if unknown:** Yes (continuous reward was a major change that needs thorough validation)

## Q2: Do you want to run adversarial tests to specifically try to "hack" or exploit the new reward system?
**Default if unknown:** Yes (important to identify potential gaming strategies before full training)

## Q3: Should the validation include comparing the new reward system against the old relative-improvement reward on sample scenarios?
**Default if unknown:** Yes (demonstrates improvement and validates the change was beneficial)

## Q4: Do you want to test the reward normalization scheme across different SCM types and variable ranges?
**Default if unknown:** Yes (ensures robustness across different problem instances)

## Q5: Should we run actual short training trials to validate that policies improve with the new system before setting up full training configuration?
**Default if unknown:** Yes (empirical validation that learning actually occurs with new rewards)