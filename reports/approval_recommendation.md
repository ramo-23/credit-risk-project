# Model Approval Recommendation

## Recommendation: **Conditional approval - address concerns**

### Identified Issues

- Calibration concerns (high Brier score)

### Conditions and Monitoring Requirements

- Implement monthly PSI monitoring for PD scores and key features
- Monitor calibration metrics and trigger recalibration if Brier > 0.20
- Review model performance quarterly with segment-level analysis
- Establish retraining triggers if AUC drops below 0.65
