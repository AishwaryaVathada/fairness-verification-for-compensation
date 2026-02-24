# Results

This directory is populated automatically when `main.py` runs. Contents are
excluded from version control (see `.gitignore`) because they are fully
reproducible from `data/Employee_Data.csv`.

## Structure

```
results/
  figures/     -- PNG visualizations (boxplots, CIs, heatmaps, etc.)
  tables/      -- CSV and JSON exports of statistical test results
```

To regenerate all outputs:

```bash
python main.py --data data/Employee_Data.csv
```
