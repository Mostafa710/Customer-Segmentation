# Customer Segmentation with Credit Card Behavioral Data

This project builds a **customer segmentation pipeline** for credit card users using exploratory analysis, preprocessing, dimensionality reduction, and unsupervised learning.

The core workflow is implemented in:
- `customer_segmentation.ipynb`

It segments customers into **7 behavior-based clusters** to support downstream use cases such as risk profiling, retention campaigns, product recommendations, and credit strategy optimization.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Modeling Approach](#modeling-approach)
- [Cluster Profiles](#cluster-profiles)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [How to Run](#how-to-run)
- [Results and Business Value](#results-and-business-value)
- [Future Improvements](#future-improvements)
- [Contact](#contact)

---

## Project Overview

Credit card customers exhibit diverse behavior in spending, cash usage, and repayment. A single engagement or risk policy is rarely optimal for all customers.

This notebook-driven project:
1. Cleans and preprocesses customer behavior data.
2. Handles missing values with domain-aware imputation.
3. Applies log transformation to reduce skewness in monetary/transactional features.
4. Uses **t-SNE** to visualize structure in the data.
5. Trains a **Gaussian Mixture Model (GMM)** to discover latent customer segments.
6. Profiles each segment using key financial and behavioral indicators.

---

## Dataset

The notebook expects a file named:
- `CC GENERAL.csv`

It is loaded via:
```python
pd.read_csv('./CC GENERAL.csv')
```

### Features used
After dropping `CUST_ID`, the analysis is performed on 17 features:

- `BALANCE`
- `BALANCE_FREQUENCY`
- `PURCHASES`
- `ONEOFF_PURCHASES`
- `INSTALLMENTS_PURCHASES`
- `CASH_ADVANCE`
- `PURCHASES_FREQUENCY`
- `ONEOFF_PURCHASES_FREQUENCY`
- `PURCHASES_INSTALLMENTS_FREQUENCY`
- `CASH_ADVANCE_FREQUENCY`
- `CASH_ADVANCE_TRX`
- `PURCHASES_TRX`
- `CREDIT_LIMIT`
- `PAYMENTS`
- `MINIMUM_PAYMENTS`
- `PRC_FULL_PAYMENT`
- `TENURE`

### Missing-value strategy
- `CREDIT_LIMIT`: imputed with median (only one missing case reported).
- `MINIMUM_PAYMENTS`: imputed with `0` after checking payment behavior (`PRC_FULL_PAYMENT`).

---

## Methodology

### 1) Data exploration and cleaning
- Checked data types and null counts.
- Investigated missing-value rows for contextual interpretation.
- Checked duplicates.
- Removed identifier column (`CUST_ID`) before modeling.

### 2) Distribution analysis
Grouped histograms were used to inspect distributions for:
- Balance-related features
- Purchase-related features
- Cash-advance features
- Payment features
- Credit limit and tenure

### 3) Feature transformation
Many monetary features were right-skewed, so log transform was applied to selected columns:
- A tiny epsilon (`1e-8`) is added before `np.log(...)` to avoid log(0).
- Frequency/percentage columns and `TENURE` were excluded from log transformation.

---

## Modeling Approach

### Dimensionality reduction for inspection
- **t-SNE** was used first on a 10% sample to inspect separability.
- A full-data t-SNE projection was later used for colored cluster visualization.

### Clustering model
- **Algorithm:** Gaussian Mixture Model (`sklearn.mixture.GaussianMixture`)
- **Number of clusters:** 7
- **Random seed:** 42

This yields soft-probabilistic clustering behavior while still assigning each customer to a final segment label.

---

## Cluster Profiles

The model identifies 7 practical customer segments:

### Cluster 0 — Cash + Transaction Revolvers
- High balance and high cash-advance behavior
- Moderate one-off purchases
- Low full-payment rate
- Typically revolving users with mixed cash/purchase usage

### Cluster 1 — Low-Activity Installment Planners
- Low balance and near-zero cash-advance activity
- Moderate installment usage
- High full-payment ratio
- Low activity, lower risk segment

### Cluster 2 — Pure Cash Borrowers
- High balance with high cash advances
- Minimal purchase behavior
- Low full-payment ratio
- Credit-risk sensitive segment

### Cluster 3 — Premium Full-Pay Users
- Strong one-off + installment purchase activity
- Higher payments and high full-payment ratio
- Strategically valuable low-risk customers

### Cluster 4 — Light One-Off Users
- Low balance and low cash-advance activity
- Moderate one-off purchasing
- Mid-level repayment discipline

### Cluster 5 — Installment-Dependent Revolvers
- High balance and high cash-advance use
- Installment reliance
- Low full-payment behavior
- Moderate-to-high risk profile

### Cluster 6 — Heavy Multi-Product Revolvers
- High activity across cash, one-off, and installment dimensions
- High payment amount but low full-payment percentage
- Potentially profitable but risk-sensitive

---

## Project Structure

```text
.
├── customer_segmentation.ipynb   # End-to-end analysis and modeling notebook
└── README.md                     # Project documentation
```

---

## Setup and Installation

### 1) Clone repository
```bash
git clone https://github.com/Mostafa710/Customer-Segmentation.git
cd Customer-Segmentation
```

### 2) Create virtual environment (recommended)
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows
```

### 3) Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### 4) Add dataset
Place `CC GENERAL.csv` in the project root so the notebook can load it.

---

## How to Run

### Option A: Jupyter Notebook
```bash
jupyter notebook
```
Then open `customer_segmentation.ipynb` and run all cells sequentially.

### Option B: JupyterLab
```bash
jupyter lab
```

---

## Results and Business Value

The segmentation output can be used to:
- Design targeted engagement campaigns by customer behavior archetype
- Personalize credit-limit and risk-control policies
- Distinguish low-risk transactors from revolving borrowers
- Optimize cross-sell and installment offers
- Prioritize retention for high-value segments

---

## Future Improvements

- Evaluate alternate clustering methods (e.g., KMeans, HDBSCAN) and compare metrics.
- Add quantitative validation (Silhouette, Davies-Bouldin, Calinski-Harabasz).
- Build a reusable training/inference pipeline (`src/` package) instead of notebook-only code.
- Export segment assignments and profile dashboards for business users.
- Add model versioning and experiment tracking.

---

## Contact
For questions or collaboration, feel free to connect:
[LinkedIn](https://www.linkedin.com/in/mostafa-mamdouh-80b110228)
