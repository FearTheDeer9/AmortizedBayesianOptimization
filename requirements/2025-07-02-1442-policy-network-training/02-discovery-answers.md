# Discovery Answers

## Q1: Should the training validation focus primarily on detecting bugs in the training loop itself?
**Answer:** Yes - focusing on policy model training step specifically, part of three-step plan (policy → surrogate → joint)

## Q2: Do you want to run actual training experiments to validate the policy learning, or just static code analysis?
**Answer:** Yes - run actual training experiments to validate policy learning

## Q3: Should the validation include testing both behavioral cloning and GRPO phases of policy training?
**Answer:** GRPO only - behavioral cloning can be handled afterwards

## Q4: Do you want to focus on intervention selection for structure learning OR target optimization, or both equally?
**Answer:** Both - validate both objectives. Sanity check: train with one reward component zeroed out to verify optimization pressure works correctly for each component

## Q5: Should the validation process include creating simplified test scenarios before running the full 3-minute validation script?
**Answer:** Yes - create as many low-level validations as beneficial before actual training