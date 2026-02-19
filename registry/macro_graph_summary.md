# QSIG Macro Graph

**Version**: 1.0.0
**Created**: 2026-01-13T00:03:50.493148Z

## Nodes

### Events
- **CPI**: Consumer Price Index release
- **CPI_PREDICTED**: St. Louis Fed expected 1-year inflation
- **UNEMPLOYMENT**: BLS Employment Situation

### Instruments
- **HY_OAS**: High-Yield OAS spread
- **VIX**: CBOE Volatility Index
- **YIELD_2Y**: 2-Year Treasury yield
- **YIELD_10Y**: 10-Year Treasury yield

## Edge Slots

### CPI->HY_OAS

CPI announcement → large HY OAS move probability

**Features**: cpi_shock_abs + ['yield_vol_10y', 'hy_vol', 'slope_10y_2y', 'fed_funds', 'hy_oas_before', 'stlfsi']

**Edges**:

| Edge ID | Model | FN Constraint | Prob Cutoff | AUC | TN/FP |
|---------|-------|---------------|-------------|-----|-------|
| CPI->HY_OAS__fn_1pct__cpi_hy_rf_shallow_w20_20bp | RF_shallow | ≤1% | 0.15 | 0.907 | 7.82x |
| CPI->HY_OAS__fn_5pct__cpi_hy_rf_shallow_w20_20bp ⭐ | RF_shallow | ≤5% | 0.15 | 0.907 | 7.82x |

### CPI->YIELD_10Y

CPI announcement → large 10Y yield move probability

**Features**: cpi_shock_abs + ['yield_vol_10y', 'slope_10y_2y', 'fed_funds', 'expinf_1y', 'y_10y_before', 'stlfsi']

**Edges**:

| Edge ID | Model | FN Constraint | Prob Cutoff | AUC | TN/FP |
|---------|-------|---------------|-------------|-----|-------|
| CPI->YIELD_10Y__fn_1pct__cpi_10y_rf_shallow_w15_20bp | RF_shallow | ≤1% | 0.15 | 0.819 | 3.47x |
| CPI->YIELD_10Y__fn_5pct__cpi_10y_rf_shallow_w15_20bp ⭐ | RF_shallow | ≤5% | 0.15 | 0.819 | 3.47x |

### CPI_PREDICTED->YIELD_10Y

Inflation expectation update → large 10Y move probability

**Features**: expinf_change + ['yield_vol_10y', 'slope_10y_2y', 'fed_funds', 'expinf_1y', 'y_10y_before', 'stlfsi']

**Edges**:

| Edge ID | Model | FN Constraint | Prob Cutoff | AUC | TN/FP |
|---------|-------|---------------|-------------|-----|-------|
| CPI_PREDICTED->YIELD_10Y__fn_1pct__cpi_predicted_10y_logreg_balanced_15bp | LogReg | ≤1% | 0.40 | 0.894 | 2.66x |
| CPI_PREDICTED->YIELD_10Y__fn_5pct__cpi_predicted_10y_logreg_balanced_15bp ⭐ | LogReg | ≤5% | 0.40 | 0.894 | 2.66x |

### UNEMP->VIX

Unemployment release → large VIX move probability

**Features**: unemp_surprise_abs + ['vix_vol', 'yield_vol_10y', 'slope_10y_2y', 'fed_funds', 'vix_before', 'stlfsi']

**Edges**:

| Edge ID | Model | FN Constraint | Prob Cutoff | AUC | TN/FP |
|---------|-------|---------------|-------------|-----|-------|
| UNEMP->VIX__fn_1pct__unemp_vix_extratrees_w20_3p5pt | ExtraTrees | ≤1% | 0.05 | 0.812 | 0.65x |
| UNEMP->VIX__fn_5pct__unemp_vix_rf_shallow_w50_1p5pt ⭐ | RF_shallow | ≤5% | 0.35 | 0.721 | 0.80x |
