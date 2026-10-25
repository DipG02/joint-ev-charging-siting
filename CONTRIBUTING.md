# Contributing

Contributions that improve reproducibility, documentation, GIS interoperability, testing, or the planning model are welcome.

For a code contribution:

1. Create a focused branch.
2. Preserve stable GIS identifiers and document any schema change.
3. Run `python tools/validate_inputs.py data/processed` if data-loading logic changes.
4. Run `python -m py_compile code/*.py tools/*.py`.
5. Test with a small, separate output directory; never overwrite reference results.
6. Describe the effective configuration, expected behavioral change, and validation evidence in the pull request.

Do not commit Gurobi license files, ArcGIS credentials, personal filesystem paths, unpublished manuscript material, proprietary geodatabases, or large generated results. Contributions that change scientific assumptions should update the relevant documentation and clearly distinguish a methodological extension from reproduction of the paper.

