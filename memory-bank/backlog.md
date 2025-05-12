# Backlog

## TODOs

- Refactor `StructuralCausalModel.do_intervention` to return a new SCM object (copy-on-intervention) rather than modifying in place, for functional purity and composability. Update all dependent code and tests to use the new pattern. (See test_random_dag_integration for current workaround.) 