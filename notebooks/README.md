# Notebooks

This directory is for exploratory Jupyter notebooks.

## Conventions

1. **Notebooks are for exploration, not production logic.**
   Any reusable code discovered during exploration should be refactored
   into the `src/` package.

2. **Naming**: Use descriptive, numbered names:
   - `01_data_exploration.ipynb`
   - `02_baseline_generation.ipynb`

3. **Reproducibility**: Each notebook should specify the experiment config
   and seed used.  Ideally, include the environment snapshot at the top.

4. **Do not commit large outputs.**  Clear cell outputs before committing,
   or use `nbstripout`.
