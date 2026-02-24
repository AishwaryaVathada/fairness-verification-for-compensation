# Employee Compensation Equity Analysis

A modular statistical analysis pipeline for evaluating fairness in
compensation practices across hierarchical levels, gender, and departments
within a multinational company (n=4,600 employees).

## Problem Statement

**Is the company equitable in its compensation practices across hierarchical
levels?**

This analysis investigates three sub-questions:

1. **Gender pay equity**: Do statistically significant salary differences exist
   between male and female employees within the same position levels?
2. **Benefits allocation**: Do Housing Allowance (HA), COLA, and CPF
   contributions vary significantly across positions in ways that may indicate
   structural inequity (particularly HA and COLA, which arguably should be
   position-agnostic)?
3. **Compensation structure**: How do the relative shares of salary components
   shift across the hierarchy, and are there interaction effects between
   position and gender?

## Methods

The pipeline applies a layered analytical approach, progressing from
descriptive diagnostics through parametric and non-parametric inference.

### Descriptive Statistics
- Extended summary statistics (mean, median, IQR, skewness, excess kurtosis,
  coefficient of variation)
- Distributional diagnostics: Shapiro-Wilk, D'Agostino-Pearson K-squared,
  Anderson-Darling normality tests
- Categorical frequency analysis with Shannon entropy
- Compensation decomposition (component shares of gross salary by position)
- Pearson and Spearman correlation analysis with Fisher z confidence intervals

### Inferential Statistics
- Parametric confidence intervals (t-based) for salary by position and gender
- BCa bootstrap confidence intervals for mean and median salary
- Normal tolerance intervals (95/95)
- Central Limit Theorem demonstration via sampling distribution simulation

### Hypothesis Testing
- **Gender pay gap**: Welch's t-test and Mann-Whitney U per position, with
  Cohen's d effect sizes and Holm-Bonferroni correction for multiplicity
- **HA / COLA / CPF across positions**: One-way ANOVA with Levene's test for
  homoscedasticity, Welch's ANOVA (robust alternative), Kruskal-Wallis H test
  (non-parametric), Tukey HSD and Games-Howell post-hoc comparisons,
  eta-squared and omega-squared effect sizes
- **Interaction effects**: Type II two-way ANOVA (Position x Gender) on salary
- **Gender-position independence**: Chi-squared test with Cramer's V
- Post-hoc power analysis for two-sample comparisons

## Project Structure

```
employee-compensation-analysis/
    config.py              -- Centralized configuration (paths, schema, parameters)
    main.py                -- Pipeline orchestrator (CLI entry point)
    requirements.txt       -- Python dependencies
    src/
        __init__.py
        data_loader.py     -- Ingestion, validation, position normalization
        descriptive.py     -- Extended summaries, normality, correlation
        inferential.py     -- CIs (parametric, bootstrap, tolerance), CLT simulation
        hypothesis_testing.py -- t-tests, ANOVA, post-hocs, effect sizes, power
        visualization.py   -- Publication-quality plotting functions
        utils.py           -- Logging, serialization, directory management
    data/
        README.md          -- Schema documentation
        Employee_Data.csv  -- (not tracked; place here before running)
    results/
        README.md
        figures/           -- Generated PNG visualizations
        tables/            -- Generated CSV/JSON result exports
    notebooks/
        emp_data_comp_analysis_codes.ipynb  -- Original exploratory notebook
```

## Setup

```bash
git clone https://github.com/<your-username>/employee-compensation-analysis.git
cd employee-compensation-analysis
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Place `Employee_Data.csv` in the `data/` directory.

## Usage

Run the full pipeline:

```bash
python main.py
```

With a custom data path:

```bash
python main.py --data /path/to/your/data.csv
```

All figures are written to `results/figures/` and tabular outputs to
`results/tables/`.

## Key Findings

Results from the IS630 course analysis (4,600 employees, 10 position levels,
6 departments):

### Gender Pay Gap
No statistically significant gender pay differences were detected within any
position level after Holm-Bonferroni correction (all adjusted p > 0.05). Effect
sizes (Cohen's d) were uniformly negligible to small. Both parametric
(Welch's t) and non-parametric (Mann-Whitney U) tests converged on the same
conclusion, strengthening confidence in the finding.

### Housing Allowance (HA)
One-way ANOVA revealed significant between-position differences
(F = 19.39, p < 0.001), but with a small effect size (eta-squared ~ 0.037).
HA varies much less across positions than COLA or CPF. This is noteworthy:
housing costs are largely determined by geography, not seniority, so the
observed (modest) position-based variation warrants review.

### COLA and CPF
Both COLA and CPF showed highly significant position effects
(F > 1200, p ~ 0.0) with large effect sizes (eta-squared > 0.70). The Tukey
HSD analysis confirmed that nearly all pairwise position comparisons are
significantly different. While this is expected for CPF (which scales with
salary), the magnitude of COLA variation is less justifiable since
cost-of-living adjustments should arguably reflect regional factors, not
individual seniority.

### Compensation Structure
Compensation decomposition reveals that the share of gross salary attributable
to COLA and CPF increases substantially with seniority, while HA share
decreases. This structural shift means that senior employees receive a
disproportionate share of benefits beyond base salary.

## Extending the Pipeline

The modular design makes it straightforward to adapt this pipeline to different
datasets or additional analyses:

- **New dataset**: Modify `config.py` to reflect column names and schema
- **New hypothesis test**: Add a function to `src/hypothesis_testing.py` and
  call it from the appropriate stage in `main.py`
- **New visualization**: Add to `src/visualization.py`; all plot functions
  follow the same `(data, ..., save_as=None) -> Figure` signature

## Technical Notes

- All hypothesis tests are conducted at alpha = 0.05 unless otherwise stated.
- Multiple comparison corrections (Holm-Bonferroni) are applied wherever
  families of tests are run across position levels.
- Bootstrap CIs use 10,000 resamples with BCa (bias-corrected and accelerated)
  method for robustness to skewness.
- Position titles are normalized from granular job titles (e.g., "Senior Data
  Engineer") to broad levels (e.g., "Senior") using keyword matching. The
  canonical hierarchy was empirically determined from salary distributions.
- Non-parametric alternatives (Mann-Whitney U, Kruskal-Wallis) are reported
  alongside parametric tests to assess sensitivity to distributional
  assumptions.

## License

MIT
