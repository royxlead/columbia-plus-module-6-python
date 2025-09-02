# Columbia+ Module 6: K-Class Bayes Classifier (Python)

This project implements a **K-class Bayes classifier** as part of Columbia+ Machine Learning I (Module 6). The classifier uses a generative Gaussian model with maximum likelihood estimation (MLE) to compute class priors and class-specific parameters, enabling accurate predictions on unseen data.

---

## Features

- Implements a **generative Bayes classifier** for multi-class problems.
- Estimates:
  - **Class prior probabilities** via maximum likelihood.
  - **Class-specific Gaussian parameters** (mean and covariance).
- Predicts test data points based on learned parameters.
- Works with datasets structured as `(X_train, y_train, X_test)`.

---

## Dataset

The project uses the provided course data files:

- `X_train.csv` → Training covariates (features).
- `y_train.csv` → Training labels (class indices from 0–9).
- `X_test.csv` → Test covariates for prediction.

Each row in `X_train.csv` corresponds to one sample, aligned with the same row in `y_train.csv`.

---

## How It Works

1. **Model Assumptions**  
   For each class `k`, the data is assumed to follow a Gaussian distribution:  
   \[
   x_i | y_i = k \sim \mathcal{N}(\mu_k, \Sigma_k)
   \]

2. **Parameter Estimation**  
   - Class priors:  
     \[
     \pi_k = \frac{\text{number of samples in class } k}{\text{total samples}}
     \]
   - Class means (`μk`) and covariances (`Σk`) are estimated from training data.

3. **Prediction**  
   For a new point `x`, the classifier computes the posterior probability for each class:  
   \[
   P(y=k|x) \propto \pi_k \cdot \mathcal{N}(x|\mu_k, \Sigma_k)
   \]  
   The predicted class is the one with the highest posterior.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/royxlead/columbia-plus-module-6-python.git
   cd columbia-plus-module-6-python

2. Set up a virtual environment (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. Install required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

1. Place the dataset files (`X_train.csv`, `y_train.csv`, `X_test.csv`) in the project directory.
2. Run the classifier script:

   ```bash
   python main.py
   ```
3. The output predictions for `X_test.csv` will be generated and saved.

---

## Directory Structure

```
columbia-plus-module-6-python/
│── X_train.csv         # Training features
│── y_train.csv         # Training labels
│── X_test.csv          # Test features
│── main.py             # Implementation of Bayes classifier
│── requirements.txt    # Dependencies
│── README.md           # Project documentation
```

---

## References

* Columbia+ Machine Learning I, Module 6 Project
* UCI Machine Learning Repository: [https://archive.ics.uci.edu/ml](https://archive.ics.uci.edu/ml)

---

## License

This project is for educational purposes only as part of Columbia+ Machine Learning I coursework.
