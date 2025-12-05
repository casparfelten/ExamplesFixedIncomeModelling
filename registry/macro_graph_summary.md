# QSIG Macro Graph

**Version**: 1.0.0
**Created**: 2025-12-05T21:54:19.883250Z

## Nodes

### Events
- **CPI**: Consumer Price Index release
- **UNEMPLOYMENT**: BLS Employment Situation

### Instruments
- **HY_OAS**: High-Yield OAS spread
- **VIX**: CBOE Volatility Index

## Edge Slots

### CPI->HY_OAS

CPI → HY_OAS large move probability

**Features**: cpi_shock_abs + ['yield_vol_10y', 'hy_vol', 'slope_10y_2y', 'fed_funds', 'hy_oas_before', 'stlfsi']

**Edges**:

| Edge ID | Model | FN Constraint | Prob Cutoff | AUC | TN/FP |
|---------|-------|---------------|-------------|-----|-------|
| CPI->HY_OAS__FN1pct__LogReg_w50_thr0p15 | LogReg | ≤1% | 0.80 | 0.841 | 1.56x |
| CPI->HY_OAS__FN5pct__LogReg_w50_thr0p15 ⭐ | LogReg | ≤5% | 0.80 | 0.841 | 1.56x |

### UNEMP->VIX

UNEMPLOYMENT → VIX large move probability

**Features**: unemp_surprise_abs + ['vix_vol', 'yield_vol_10y', 'slope_10y_2y', 'fed_funds', 'vix_before', 'stlfsi']

**Edges**:

| Edge ID | Model | FN Constraint | Prob Cutoff | AUC | TN/FP |
|---------|-------|---------------|-------------|-----|-------|
| UNEMP->VIX__FN1pct__RF_shallow_w50_thr2p50 | RF_shallow | ≤1% | 0.20 | 0.777 | 2.41x |
| UNEMP->VIX__FN5pct__GB_w20_thr2p00 ⭐ | GB | ≤5% | 0.05 | 0.693 | 6.86x |
