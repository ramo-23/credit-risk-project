# PD Banding Analysis Report

**Generated:** 2026-02-05 19:43:04
**Source Data:** `data\processed\scored_validation.csv`
**Number of Bands:** 10

## Executive Summary

- **Total Observations:** 426,149
- **Overall Default Rate:** 1.38%
- **PD Score Range:** [0.0000, 1.0000]
- **Bands Created:** 10
- **Average Band Size:** 42615 observations
- **Mean Calibration Gap:** 0.3690

## Band Performance Summary

| Band | PD Range | Observed Default Rate | Population Share | Count |
|------|----------|----------------------|------------------|-------|
| 0 | [0.0000, 0.0175] | 0.06% | 10.00% | 42,615 |
| 1 | [0.0175, 0.0493] | 0.05% | 10.00% | 42,615 |
| 2 | [0.0493, 0.0985] | 0.08% | 10.00% | 42,615 |
| 3 | [0.0985, 0.1713] | 0.11% | 10.00% | 42,615 |
| 4 | [0.1713, 0.2744] | 0.11% | 10.00% | 42,615 |
| 5 | [0.2744, 0.4158] | 0.19% | 10.00% | 42,614 |
| 6 | [0.4158, 0.5900] | 0.27% | 10.00% | 42,615 |
| 7 | [0.5900, 0.7797] | 0.47% | 10.00% | 42,615 |
| 8 | [0.7798, 0.9312] | 1.43% | 10.00% | 42,615 |
| 9 | [0.9312, 1.0000] | 11.05% | 10.00% | 42,615 |

## Detailed Band Analysis

### Minimal Risk (Band 0)

- **PD Range:** [0.000000, 0.017459]
- **Observed Default Rate:** 0.0587%
- **Population Share:** 10.00%
- **Count:** 42,615 observations
- **Calibration:** Well calibrated ✓

### Very Low Risk (Band 1)

- **PD Range:** [0.017460, 0.049327]
- **Observed Default Rate:** 0.0516%
- **Population Share:** 10.00%
- **Count:** 42,615 observations
- **Calibration:** Over-predicting risk (observed 3.29% lower)

### Low Risk (Band 2)

- **PD Range:** [0.049327, 0.098482]
- **Observed Default Rate:** 0.0774%
- **Population Share:** 10.00%
- **Count:** 42,615 observations
- **Calibration:** Over-predicting risk (observed 7.31% lower)

### Low-Medium Risk (Band 3)

- **PD Range:** [0.098482, 0.171272]
- **Observed Default Rate:** 0.1056%
- **Population Share:** 10.00%
- **Count:** 42,615 observations
- **Calibration:** Over-predicting risk (observed 13.38% lower)

### Medium Risk (Band 4)

- **PD Range:** [0.171273, 0.274357]
- **Observed Default Rate:** 0.1126%
- **Population Share:** 10.00%
- **Count:** 42,615 observations
- **Calibration:** Over-predicting risk (observed 22.17% lower)

### Medium-High Risk (Band 5)

- **PD Range:** [0.274360, 0.415764]
- **Observed Default Rate:** 0.1854%
- **Population Share:** 10.00%
- **Count:** 42,614 observations
- **Calibration:** Over-predicting risk (observed 34.32% lower)

### High Risk (Band 6)

- **PD Range:** [0.415770, 0.590003]
- **Observed Default Rate:** 0.2746%
- **Population Share:** 10.00%
- **Count:** 42,615 observations
- **Calibration:** Over-predicting risk (observed 50.01% lower)

### Very High Risk (Band 7)

- **PD Range:** [0.590016, 0.779749]
- **Observed Default Rate:** 0.4693%
- **Population Share:** 10.00%
- **Count:** 42,615 observations
- **Calibration:** Over-predicting risk (observed 68.02% lower)

### Critical Risk (Band 8)

- **PD Range:** [0.779753, 0.931219]
- **Observed Default Rate:** 1.4267%
- **Population Share:** 10.00%
- **Count:** 42,615 observations
- **Calibration:** Over-predicting risk (observed 84.12% lower)

### Extreme Risk (Band 9)

- **PD Range:** [0.931219, 1.000000]
- **Observed Default Rate:** 11.0454%
- **Population Share:** 10.00%
- **Count:** 42,615 observations
- **Calibration:** Over-predicting risk (observed 85.52% lower)

## Recommended Actions by Band

- **Minimal Risk** (Band 0): **Auto-approve** - Minimal risk, standard terms
- **Very Low Risk** (Band 1): **Auto-approve** - Minimal risk, standard terms
- **Low Risk** (Band 2): **Auto-approve** - Minimal risk, standard terms
- **Low-Medium Risk** (Band 3): **Auto-approve** - Minimal risk, standard terms
- **Medium Risk** (Band 4): **Auto-approve** - Minimal risk, standard terms
- **Medium-High Risk** (Band 5): **Auto-approve** - Minimal risk, standard terms
- **High Risk** (Band 6): **Auto-approve** - Minimal risk, standard terms
- **Very High Risk** (Band 7): **Auto-approve** - Minimal risk, standard terms
- **Critical Risk** (Band 8): **Auto-approve** with monitoring - Low risk, consider favorable terms
- **Extreme Risk** (Band 9): **Manual review required** - Enhanced due diligence, risk-based pricing

## Implementation Guidelines

### Model Monitoring
- Review band performance monthly to detect distribution shifts
- Monitor actual vs predicted default rates for calibration drift
- Recalibrate bands quarterly or when performance degrades

### Business Integration
- Map bands to existing credit policies and approval workflows
- Set override authorities for borderline cases
- Document exceptions and track override performance
- Integrate with pricing strategy based on risk-adjusted returns

### Regulatory Compliance
- Ensure consistent application across protected classes
- Document business justification for band thresholds
- Maintain audit trail of band definitions and changes
- Prepare adverse action notices for declined applications

## Appendix

### Methodology
- Bands created using quantile-based segmentation (n=10)
- Observed default rates calculated from validation dataset
- Population shares represent proportion of total observations
- Risk labels assigned based on observed default rate thresholds
