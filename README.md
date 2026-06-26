# Calibration is All You Need? When Post-hoc Calibration Helps, Hurts, and Misleads in Clinical Risk Prediction

A systematic multi-dataset investigation of post-hoc calibration methods across clinical risk prediction tasks.

## Key Finding

Post-hoc calibration is **not universally beneficial**. Across 6 clinical datasets and 6 model families, we show that calibration degrades performance in a significant fraction of cases — and provide diagnostic criteria for when to apply which method.

## Experiment Design

- **6 Datasets**: Pima Diabetes, Cleveland Heart Disease, Breast Cancer, Indian Liver, Early Diabetes, Chronic Kidney Disease
- **6 Models**: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, SVM (RBF), MLP
- **4 Calibration Methods**: Uncalibrated, Platt Scaling, Isotonic Regression, Beta Calibration
- **Evaluation**: 10-fold stratified CV, paired t-tests for statistical significance
- **Metrics**: AUC-ROC, ECE, MCE, Brier Score, Calibration Slope, Log Loss

## Reproducing

```bash
pip install -r requirements.txt
python run_experiments.py      # ~10 min on CPU
python generate_figures.py     # produces figures/
```

## Citation

```bibtex
@article{nanda2026calibration,
  title={Calibration is All You Need? When Post-hoc Calibration Helps, Hurts, and Misleads in Clinical Risk Prediction},
  author={Nanda, Siddhardha},
  year={2026}
}
```

## License

MIT
