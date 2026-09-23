Trigger when:
- adding LIBERO tasks
- changing LIBERO environment
- changing camera configuration
- changing robot/controller
- validating task assets
- reproducing benchmark behavior

Workflow:
1. Identify official source.
2. Pin revision.
3. Inspect existing official implementation.
4. Do not reconstruct resource manually.
5. Run environment integrity checks.
6. Run official demonstration replay if available.
7. Run one-task smoke test.
8. Run quantitative evaluation.
9. Save provenance.

Never:
- create replacement CAD
- alter BDDL
- randomly reposition official initial state
- substitute robot model
- alter camera geometry
- silently change control frequency