# QSIG Macro Graph

**Version**: 1.0.0
**Created**: 2025-12-05T22:18:23.030012Z

## Nodes

### Events
- **CPI**: Consumer Price Index release
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

### UNEMP->VIX

Unemployment release → large VIX move probability

**Features**: unemp_surprise_abs + ['vix_vol', 'yield_vol_10y', 'slope_10y_2y', 'fed_funds', 'vix_before', 'stlfsi']

**Edges**:

| Edge ID | Model | FN Constraint | Prob Cutoff | AUC | TN/FP |
|---------|-------|---------------|-------------|-----|-------|
| UNEMP->VIX__fn_1pct__unemp_vix_extratrees_w20_3p5pt | ExtraTrees | ≤1% | 0.05 | 0.812 | 0.65x |
| UNEMP->VIX__fn_5pct__unemp_vix_rf_shallow_w50_1p5pt ⭐ | RF_shallow | ≤5% | 0.35 | 0.721 | 0.80x |
