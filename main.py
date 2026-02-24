"""
main.py
-------
Orchestrates the full compensation analysis pipeline.

Usage:
    python main.py                          # run with default data path
    python main.py --data path/to/data.csv  # custom data path

Pipeline stages:
    1. Data loading and preprocessing
    2. Descriptive statistics and distributional diagnostics
    3. Inferential statistics (confidence intervals, bootstrap)
    4. Hypothesis testing (gender pay gap, ANOVA on benefits, interaction)
    5. Results export (tables, figures)
"""

from __future__ import annotations

import argparse
import logging
import warnings

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CI/headless runs
import matplotlib.pyplot as plt

import config as cfg
from src.data_loader import load_and_prepare, schema_check, load_csv, missing_value_report
from src.descriptive import (
    extended_describe,
    normality_tests,
    categorical_summary,
    compensation_decomposition,
    correlation_matrix,
    pairwise_correlation_test,
    grouped_correlation,
)
from src.inferential import (
    grouped_ci,
    bootstrap_ci,
    simulate_sampling_distribution,
    normal_tolerance_interval,
)
from src.hypothesis_testing import (
    gender_pay_gap_by_position,
    one_way_anova,
    welch_anova,
    kruskal_wallis,
    games_howell,
    two_way_anova,
    chi_squared_independence,
    check_normality_per_group,
    check_homogeneity_of_variance,
)
from src.visualization import (
    plot_categorical_distributions,
    plot_numeric_distributions,
    plot_boxplot_by_group,
    plot_violin_by_group_and_hue,
    plot_grouped_ci,
    plot_correlation_heatmap,
    plot_bootstrap_distribution,
    plot_pay_gap_results,
    plot_compensation_decomposition,
)
from src.utils import (
    setup_logging,
    ensure_dirs,
    save_dataframe,
    save_result_dict,
    print_section,
)


warnings.filterwarnings("ignore", category=FutureWarning)
logger = setup_logging()


def parse_args():
    parser = argparse.ArgumentParser(description="Employee Compensation Analysis")
    parser.add_argument(
        "--data", type=str, default=str(cfg.DATA_FILE),
        help="Path to the employee data CSV file",
    )
    return parser.parse_args()


# ===================================================================
# STAGE 1 : DATA LOADING & PREPROCESSING
# ===================================================================

def stage_data_loading(data_path: str):
    print_section("STAGE 1: DATA LOADING & PREPROCESSING")

    # Raw diagnostics first
    raw_df = load_csv(data_path)
    diag = schema_check(raw_df)
    logger.info("Raw shape: %s", diag["shape"])
    logger.info("Duplicated rows: %d", diag["duplicated_rows"])
    if not diag["missing_summary"].empty:
        logger.info("Missing values:\n%s", diag["missing_summary"])
        save_dataframe(diag["missing_summary"], "missing_value_report")

    # Full preprocessing
    df = load_and_prepare(data_path)
    logger.info("Clean shape: %s", df.shape)
    return df


# ===================================================================
# STAGE 2 : DESCRIPTIVE STATISTICS
# ===================================================================

def stage_descriptive(df):
    print_section("STAGE 2: DESCRIPTIVE STATISTICS")

    # 2a. Extended numeric summary
    num_cols = [c for c in cfg.NUMERIC_COLS if c in df.columns]
    desc = extended_describe(df, cols=num_cols)
    print(desc.to_string())
    save_dataframe(desc, "extended_describe")

    # 2b. Normality tests
    norm = normality_tests(df, cols=num_cols)
    print("\nNormality Tests:")
    print(norm[["n", "shapiro_stat", "shapiro_p", "shapiro_reject",
                "anderson_stat", "anderson_reject"]].to_string())
    save_dataframe(norm, "normality_tests")

    # 2c. Categorical summaries
    cat_cols = [c for c in cfg.CATEGORICAL_COLS if c in df.columns]
    cat_summaries = categorical_summary(df, cols=cat_cols)
    for col, tbl in cat_summaries.items():
        print(f"\n{col} (entropy={tbl.attrs['entropy']:.2f}, max={tbl.attrs['max_entropy']:.2f}):")
        print(tbl.to_string())
        save_dataframe(tbl, f"categorical_{col.lower().replace(' ', '_')}")

    # 2d. Compensation decomposition by position
    try:
        decomp = compensation_decomposition(df, group_col="Position")
        print("\nCompensation Decomposition:")
        print(decomp.to_string())
        save_dataframe(decomp, "compensation_decomposition")
        plot_compensation_decomposition(decomp, save_as="compensation_decomposition.png")
    except ValueError as e:
        logger.warning("Skipping compensation decomposition: %s", e)

    # 2e. Correlation analysis
    comp_cols = [c for c in cfg.COMPENSATION_COMPONENTS if c in df.columns]
    if len(comp_cols) >= 2:
        corr = correlation_matrix(df, cols=comp_cols)
        save_dataframe(corr, "correlation_matrix")
        plot_correlation_heatmap(corr, title="Compensation Components Correlation",
                                save_as="correlation_heatmap.png")

        # Deep dive: COLA vs CPF
        if "COLA" in df.columns and "CPF" in df.columns:
            cola_cpf = pairwise_correlation_test(df, "COLA", "CPF")
            print(f"\nCOLA-CPF Correlation: r={cola_cpf['r']:.4f}, p={cola_cpf['p_value']:.2e}")
            save_result_dict(cola_cpf, "cola_cpf_correlation")

            grp_corr = grouped_correlation(df, "COLA", "CPF", "Position")
            save_dataframe(grp_corr, "cola_cpf_correlation_by_position")

    # 2f. Visualizations
    plot_categorical_distributions(df, cat_cols, save_prefix="cat_dist")
    plot_numeric_distributions(df, num_cols[:4], save_prefix="num_dist")

    for var in ["Salary", "HA", "COLA", "CPF"]:
        if var in df.columns:
            plot_boxplot_by_group(
                df, var, "Position", order=cfg.POSITION_ORDER,
                title=f"{var} Distribution by Position",
                save_as=f"boxplot_{var.lower()}_by_position.png",
            )

    if "Gender" in df.columns and "Salary" in df.columns:
        plot_violin_by_group_and_hue(
            df, "Salary", "Position", "Gender",
            order=cfg.POSITION_ORDER,
            title="Salary Distribution by Position and Gender",
            save_as="violin_salary_gender_position.png",
        )

    plt.close("all")
    return desc


# ===================================================================
# STAGE 3 : INFERENTIAL STATISTICS
# ===================================================================

def stage_inferential(df):
    print_section("STAGE 3: INFERENTIAL STATISTICS")

    # 3a. Parametric CIs for salary by position and gender
    if "Position" in df.columns and "Salary" in df.columns:
        ci_salary = grouped_ci(df, "Salary", "Position", "Gender")
        print("Salary CIs by Position x Gender:")
        print(ci_salary.to_string())
        save_dataframe(ci_salary, "ci_salary_position_gender")
        plot_grouped_ci(
            ci_salary, "Position", "Gender",
            value_label="Salary ($)",
            title="95% CI for Mean Salary by Position and Gender",
            save_as="ci_salary_position_gender.png",
        )

    # 3b. Bootstrap CI for overall mean salary
    if "Salary" in df.columns:
        boot = bootstrap_ci(df["Salary"].dropna().values, statistic="mean", method="bca")
        print(f"\nBootstrap BCa CI for mean salary: "
              f"[{boot['ci_low']:.2f}, {boot['ci_high']:.2f}] "
              f"(observed={boot['observed']:.2f}, se={boot['bootstrap_se']:.2f})")
        save_result_dict(boot, "bootstrap_salary_mean")
        plot_bootstrap_distribution(
            boot, title="Bootstrap Distribution of Mean Salary (BCa)",
            save_as="bootstrap_salary_mean.png",
        )

        # Bootstrap CI for median salary (robust measure)
        boot_med = bootstrap_ci(df["Salary"].dropna().values, statistic="median", method="bca")
        print(f"Bootstrap BCa CI for median salary: "
              f"[{boot_med['ci_low']:.2f}, {boot_med['ci_high']:.2f}]")
        save_result_dict(boot_med, "bootstrap_salary_median")

    # 3c. Tolerance interval
    if "Salary" in df.columns:
        tol = normal_tolerance_interval(df["Salary"].dropna().values)
        print(f"\n95/95 Tolerance Interval for Salary: "
              f"[{tol['tol_low']:.2f}, {tol['tol_high']:.2f}]")
        save_result_dict(tol, "tolerance_interval_salary")

    # 3d. CIs for HA and COLA by position
    for var in ["HA", "COLA"]:
        if var in df.columns:
            ci_var = grouped_ci(df, var, "Position")
            save_dataframe(ci_var, f"ci_{var.lower()}_position")

    # 3e. Sampling distribution demonstration (CLT)
    if "Salary" in df.columns:
        samp = simulate_sampling_distribution(
            df["Salary"].dropna().values, sample_size=100, n_samples=5000,
        )
        print(f"\nSampling Distribution (n=100, 5000 draws):")
        print(f"  Empirical SE = {samp['empirical_se']:.2f}, "
              f"Theoretical SE = {samp['theoretical_se']:.2f}, "
              f"Ratio = {samp['se_ratio']:.4f}")
        print(f"  Normality of sampling dist (Shapiro p) = {samp['normality_p']:.4f}")

    plt.close("all")


# ===================================================================
# STAGE 4 : HYPOTHESIS TESTING
# ===================================================================

def stage_hypothesis_testing(df):
    print_section("STAGE 4: HYPOTHESIS TESTING")

    # ------------------------------------------------------------------
    # H1: Gender pay gap within positions
    # ------------------------------------------------------------------
    print_section("H1: Gender Pay Equity Across Positions")

    if "Gender" in df.columns and "Salary" in df.columns:
        gap_results = gender_pay_gap_by_position(df)
        print(gap_results[[
            "position", "n_male", "n_female", "raw_gap", "pct_gap",
            "welch_p", "cohens_d", "effect_size",
            "welch_p_adjusted", "reject_adjusted",
        ]].to_string(index=False))
        save_dataframe(gap_results, "gender_pay_gap_results")
        plot_pay_gap_results(gap_results, save_as="gender_pay_gap.png")

    # ------------------------------------------------------------------
    # H2: HA differences across positions (ANOVA + robust alternatives)
    # ------------------------------------------------------------------
    print_section("H2: Housing Allowance Across Positions")

    if "HA" in df.columns and "Position" in df.columns:
        # Assumption checks
        norm_ha = check_normality_per_group(df, "HA", "Position")
        print("Normality by group (HA):")
        print(norm_ha.to_string())
        save_dataframe(norm_ha, "normality_ha_by_position")

        # Classic ANOVA
        anova_ha = one_way_anova(df, "HA", "Position")
        print(f"\nOne-way ANOVA: F={anova_ha['F_statistic']:.3f}, "
              f"p={anova_ha['p_value']:.2e}, "
              f"eta^2={anova_ha['eta_squared']:.4f}, "
              f"omega^2={anova_ha['omega_squared']:.4f}")
        print(f"  Levene p={anova_ha['levene_p']:.4f} "
              f"({'equal' if anova_ha['equal_variance'] else 'UNEQUAL'} variance)")
        save_result_dict(anova_ha, "anova_ha")
        if "tukey_summary" in anova_ha:
            save_dataframe(anova_ha["tukey_summary"], "tukey_ha")

        # Welch ANOVA (robust to heteroscedasticity)
        welch_ha = welch_anova(df, "HA", "Position")
        print(f"  Welch ANOVA: stat={welch_ha.get('F_statistic', welch_ha.get('statistic', 'N/A'))}, "
              f"p={welch_ha['p_value']:.2e}")
        save_result_dict(welch_ha, "welch_anova_ha")

        # Kruskal-Wallis (non-parametric)
        kw_ha = kruskal_wallis(df, "HA", "Position")
        print(f"  Kruskal-Wallis: H={kw_ha['H_statistic']:.3f}, "
              f"p={kw_ha['p_value']:.2e}, eps^2={kw_ha['epsilon_squared']:.4f}")
        save_result_dict(kw_ha, "kruskal_wallis_ha")

        # Games-Howell post-hoc (no equal variance assumption)
        gh_ha = games_howell(df, "HA", "Position")
        save_dataframe(gh_ha, "games_howell_ha")

    # ------------------------------------------------------------------
    # H3: COLA differences across positions
    # ------------------------------------------------------------------
    print_section("H3: COLA Across Positions")

    if "COLA" in df.columns and "Position" in df.columns:
        anova_cola = one_way_anova(df, "COLA", "Position")
        print(f"One-way ANOVA: F={anova_cola['F_statistic']:.3f}, "
              f"p={anova_cola['p_value']:.2e}, "
              f"eta^2={anova_cola['eta_squared']:.4f}, "
              f"omega^2={anova_cola['omega_squared']:.4f}")
        save_result_dict(anova_cola, "anova_cola")
        if "tukey_summary" in anova_cola:
            save_dataframe(anova_cola["tukey_summary"], "tukey_cola")

        kw_cola = kruskal_wallis(df, "COLA", "Position")
        print(f"  Kruskal-Wallis: H={kw_cola['H_statistic']:.3f}, p={kw_cola['p_value']:.2e}")
        save_result_dict(kw_cola, "kruskal_wallis_cola")

        gh_cola = games_howell(df, "COLA", "Position")
        save_dataframe(gh_cola, "games_howell_cola")

    # ------------------------------------------------------------------
    # H4: CPF differences across positions
    # ------------------------------------------------------------------
    print_section("H4: CPF Across Positions")

    if "CPF" in df.columns and "Position" in df.columns:
        anova_cpf = one_way_anova(df, "CPF", "Position")
        print(f"One-way ANOVA: F={anova_cpf['F_statistic']:.3f}, "
              f"p={anova_cpf['p_value']:.2e}, "
              f"eta^2={anova_cpf['eta_squared']:.4f}")
        save_result_dict(anova_cpf, "anova_cpf")
        if "tukey_summary" in anova_cpf:
            save_dataframe(anova_cpf["tukey_summary"], "tukey_cpf")

    # ------------------------------------------------------------------
    # H5: Two-way ANOVA - Position x Gender interaction on Salary
    # ------------------------------------------------------------------
    print_section("H5: Position x Gender Interaction on Salary")

    if all(c in df.columns for c in ["Salary", "Position", "Gender"]):
        # Filter to Male/Female only for cleaner interaction analysis
        df_mf = df[df["Gender"].isin(["Male", "Female"])].copy()
        try:
            two_way = two_way_anova(df_mf, "Salary", "Position", "Gender")
            print(two_way.to_string())
            save_dataframe(two_way, "two_way_anova_salary")
        except Exception as e:
            logger.warning("Two-way ANOVA failed: %s", e)

    # ------------------------------------------------------------------
    # H6: Chi-squared - Gender independence from Position
    # ------------------------------------------------------------------
    print_section("H6: Gender x Position Independence")

    if "Gender" in df.columns and "Position" in df.columns:
        chi2_res = chi_squared_independence(df, "Gender", "Position")
        print(f"Chi-squared: chi2={chi2_res['chi2_statistic']:.3f}, "
              f"p={chi2_res['p_value']:.4f}, "
              f"Cramer's V={chi2_res['cramers_v']:.4f}")
        save_result_dict(chi2_res, "chi_squared_gender_position")

    plt.close("all")


# ===================================================================
# MAIN
# ===================================================================

def main():
    args = parse_args()
    ensure_dirs()

    logger.info("Starting Employee Compensation Analysis Pipeline")
    logger.info("Data source: %s", args.data)

    df = stage_data_loading(args.data)
    stage_descriptive(df)
    stage_inferential(df)
    stage_hypothesis_testing(df)

    print_section("PIPELINE COMPLETE")
    logger.info("Results saved to: %s", cfg.RESULTS_DIR)
    logger.info("Figures saved to: %s", cfg.FIGURES_DIR)
    logger.info("Tables saved to: %s", cfg.TABLES_DIR)


if __name__ == "__main__":
    main()
