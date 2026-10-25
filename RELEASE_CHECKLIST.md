# Release checklist

Use this short checklist when creating a tagged paper release:

- [ ] Confirm the paper citation, author names, and contact information.
- [ ] Run `python tools/validate_inputs.py data/processed`.
- [ ] Run `python examples/baseline_p200/verify_results.py`.
- [ ] Run `sha256sum -c SHA256SUMS.txt`.
- [ ] Confirm that no credentials, Gurobi license files, personal paths, or large generated outputs are committed.
- [ ] Preserve the source attributions in `docs/PUBLIC_DATA_SOURCES.md` and `DATA_LICENSE.md`.
- [ ] Add the final GitHub URL to `CITATION.cff`.
- [ ] Create a versioned GitHub release matching the paper's code and inputs.

For the patent timeline, retain the provisional filing date and the deadline for any corresponding nonprovisional or international filing with the project records.
