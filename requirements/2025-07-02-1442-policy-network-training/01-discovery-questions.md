# Discovery Questions

## Q1: Should the training validation focus primarily on detecting bugs in the training loop itself?
**Default if unknown:** Yes (ensuring correct GRPO implementation and reward propagation is fundamental)

## Q2: Do you want to run actual training experiments to validate learning, or just static analysis of the code?
**Default if unknown:** Yes (run actual training experiments to see if policies improve)

## Q3: Should the validation include testing both behavioral cloning and GRPO phases of training?
**Default if unknown:** Yes (both phases are mentioned and need validation)

## Q4: Do you want to focus on intervention selection for structure learning OR target optimization, or both equally?
**Default if unknown:** Both equally (both tasks mentioned as relevant)

## Q5: Should the validation process include creating simplified test scenarios before full training?
**Default if unknown:** Yes (safer to validate on simple cases first before comprehensive training)