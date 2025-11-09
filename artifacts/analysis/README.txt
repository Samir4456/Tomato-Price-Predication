Analysis artifacts generated on: 2025-11-09T08:34:04.491040

Files:
 - holdout_performance.csv : model metrics on holdout
 - perm_importance/*.csv : permutation importance per model
 - ablation_results.csv : ablation test results dropping feature groups
 - shap/* : shap values and images (if shap installed)
 - topk/*.csv : top-k error analysis per model
 - plots/*.html : interactive plotly plots for Actual vs Predicted and distributions

Notes:
 - SHAP is preferred for global+local model explainability; waterfall is a local decomposition (single sample).
 - LIME can be used for local explanations but SHAP gives consistent global attribution and faster tree explainer for tree models.
