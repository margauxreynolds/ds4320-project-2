# DS 4320 Project 2: Detecting Credit Card Fraud with Logistic Regression

This repository contains a complete fraud detection pipeline built on a dataset of 284,807 real credit card transactions from European cardholders. The project uses MongoDB as the document database, logistic regression as the classification model, and addresses the severe class imbalance inherent in fraud data. Materials include a Jupyter notebook pipeline, a press release, full metadata documentation, and background readings on fraud detection in financial systems.

| Spec | Value |
|:---|:---|
| Name | Margaux Reynolds |
| NetID | tsh3ut |
| DOI | [https://doi.org/10.5281/zenodo.19698842](https://doi.org/10.5281/zenodo.19698842) |
| Press Release | [Can Your Bank Tell When You're Being Robbed?](press-release.md) |
| Pipeline | [Pipeline File](pipeline/pipeline.ipynb) |
| License | [MIT](LICENSE) |

---

## Problem Definition

**General problem:** Detecting credit card fraud

**Specific problem:** Classify individual credit card transactions as fraudulent or legitimate using transaction amount, elapsed time, and 28 PCA-transformed features derived from anonymized European cardholder data.

### Motivation

Credit card fraud costs consumers and financial institutions billions of dollars every year, and most people have either experienced it personally or know someone who has. Despite how common it is, fraud detection systems still struggle to catch fraudulent transactions without also flagging legitimate ones. A reliable transaction-level model would help banks respond faster, reduce financial losses, and minimize the frustration of having a legitimate purchase declined. This project uses a real-world dataset to build a model that can distinguish fraudulent transactions from genuine ones.

### Rationale

The general problem of detecting credit card fraud can be approached in many ways, including account takeover detection, identity verification, or chargeback analysis. Narrowing to transaction-level binary classification makes the problem tractable because a well-established public dataset exists with clearly labeled outcomes. The Kaggle Credit Card Fraud Detection dataset contains 284,807 real anonymized transactions from European cardholders in September 2013, large enough to train a meaningful model while remaining manageable. Logistic regression is a natural fit because the outcome is binary and the model is interpretable enough to reason about which features drive a given prediction, an important property in a domain where explaining a fraud flag to a customer matters.

### Press Release Headline and Link

[Can Your Bank Tell When You're Being Robbed? A Data-Driven Look at Credit Card Fraud Detection](press-release.md)

---

## Domain Exposition

This project lives in the domain of financial technology and fraud prevention. Credit card fraud is a large and growing problem affecting consumers, banks, and payment networks worldwide. Financial institutions invest heavily in automated systems that evaluate transactions in real time and flag suspicious activity before it causes harm. These systems rely on machine learning models trained on historical transaction data to learn the patterns that distinguish fraudulent behavior from normal spending. The challenge is not just building an accurate model but doing so on data where fraud cases are extremely rare, which requires careful handling of class imbalance and thoughtful evaluation metrics beyond simple accuracy.

### Terminology
 
| Term | Definition |
|:---|:---|
| Fraud | A transaction intentionally made by someone other than the authorized cardholder |
| Class | Binary target variable: 0 = legitimate, 1 = fraudulent |
| Class imbalance | When one outcome is much rarer than the other in a dataset |
| PCA | Principal Component Analysis, a dimensionality reduction technique that projects data onto a new set of axes that capture the most variance |
| Logistic regression | A classification model that predicts the probability of a binary outcome |
| Precision | Of all transactions flagged as fraud, the proportion that were actually fraud |
| Recall | Of all actual fraud cases, the proportion the model correctly identified |
| False positive | A legitimate transaction incorrectly flagged as fraudulent |
| False negative | A fraudulent transaction the model missed |

### Background Readings

[OneDrive Folder](https://myuva-my.sharepoint.com/:f:/g/personal/tsh3ut_virginia_edu/IgDoURsNRFGCQ71W3oY9HvZQAdH51juKrIehheiGwwT6bjw)

| # | Title | Description | Link |
|:--|:------|:------------|:-----|
| 1 | A Supervised Machine Learning Algorithm for Detecting and Predicting Fraud in Credit Card Transactions | Peer-reviewed study comparing logistic regression, decision tree, and random forest on a simulated credit card dataset; random forest achieved 96% accuracy and 98.9% AUC. | [link](https://myuva-my.sharepoint.com/:b:/g/personal/tsh3ut_virginia_edu/IQC8AYTNHZA3Q4Tu7QtunMguAeadVmN0GD5LQ8bwa7GncZQ) |
| 2 | How Common Is Credit Card Fraud? | Experian overview of credit card fraud prevalence, types, and 2024 FTC statistics (~449K reports); provides real-world domain motivation. | [link](https://myuva-my.sharepoint.com/:b:/g/personal/tsh3ut_virginia_edu/IQDfSmFVuLQQRYThzWJ57hdcAYz6TH-Iy3ETDGDYObOX-KE) |
| 3 | Strategies for Data Imbalance in Fraud Classifiers | Practitioner blog post explaining why class imbalance is the core challenge in fraud classification and comparing SMOTE, undersampling, and SMOTE+ENN approaches. | [link](https://myuva-my.sharepoint.com/:b:/g/personal/tsh3ut_virginia_edu/IQBzHlH-WSgORLQ0VVvhEGisAfyN5MiajbQzsjQPcIt2eqo) |
| 4 | Imbalanced Classification in Fraud Detection | Medium article walking through precision-recall tradeoffs and balancing algorithm results across six classifiers on a real Kaggle fraud dataset. | [link](https://myuva-my.sharepoint.com/:b:/g/personal/tsh3ut_virginia_edu/IQANKKjcP5eCSaOtmsjMU3wuARa7XHa0-57u1hfesPUhkCU) |
| 5 | Addressing Digital Finance Challenges: The Application and Optimization of Logistic Regression Models in Fraud Detection Systems | Conference paper applying logistic regression to two real-world fraud datasets; covers feature engineering, SMOTE, regularization, and evaluation metrics. | [link](https://myuva-my.sharepoint.com/:b:/g/personal/tsh3ut_virginia_edu/IQB3Qa-JNH64Rpi8ZD8RCb_8AerkDHqIIr3LNYKk0OIgc2s) |

---

## Data Creation

### Provenance
The dataset originates from a research collaboration between Worldline and the Machine Learning Group at Université Libre de Bruxelles. It contains 284,807 credit card transactions made by European cardholders over two days in September 2013, of which 492 are fraudulent (0.172% of all transactions). Because the underlying transaction data is sensitive, all original features were anonymized using PCA, producing 28 numerical components (`V1`–`V28`). Only `Time` and `Amount` were left untransformed: `Time` is the seconds elapsed between each transaction and the first transaction in the dataset, and `Amount` is the transaction amount. Each transaction carries a binary `Class` label (`1` = fraud, `0` = legitimate), applied by the original researchers. The dataset is publicly available on Kaggle under the Open Database License and was downloaded directly from there as `creditcard.csv`.

### Code

| File | Description | Link |
|:-----|:------------|:-----|
| `data_acquisition.ipynb` | Loads `creditcard.csv`, converts rows to documents, and inserts into MongoDB Atlas | [pipeline/data_acquisition.ipynb](pipeline/data_acquisition.ipynb) |
| `pipeline.ipynb` | Queries MongoDB, trains logistic regression model, evaluates with precision/recall, and generates visualizations | [pipeline/pipeline.ipynb](pipeline/pipeline.ipynb) |

### Bias Identification
The dataset reflects only European cardholders over a 48-hour window in September 2013, which limits geographic and temporal generalizability. Because the original features are PCA-transformed and anonymized, it is impossible to assess whether the data overrepresents or underrepresents any cardholder demographic. The extreme class imbalance (0.17% fraud) also reflects real-world conditions but means a naive model will be heavily biased toward predicting the majority class.

### Bias Mitigation
Class imbalance is addressed during model training using `class_weight='balanced'` in scikit-learn, which adjusts the logistic regression loss function to penalize missed fraud cases more heavily proportional to their rarity in the training data. Model performance is evaluated using precision, recall, and AUPRC rather than accuracy, which would otherwise give a misleadingly high score on an imbalanced dataset. However, these steps do not fully resolve the bias introduced by the dataset's limited scope. The model is trained exclusively on European cardholders over a 48-hour window in 2013, so its learned decision boundary may not generalize to other regions, time periods, or spending patterns. The PCA anonymization prevents any demographic audit of the training data, meaning it is impossible to verify whether the model performs equally across different cardholder groups. These limitations should be acknowledged when interpreting model results.

### Rationale for Critical Decisions
Several judgment calls shape this pipeline. First, storing data in MongoDB suits the structure of a transaction record, which is naturally self-contained: each document represents one transaction and requires no joins. Second, logistic regression was chosen because the outcome is binary and the model is interpretable. A logistic model produces probability scores that map directly onto a fraud/not-fraud decision threshold, which can be adjusted depending on how a bank wants to trade off false positives against false negatives. Third, `class_weight='balanced'` was used rather than SMOTE to avoid introducing synthetic data points into a dataset already intended to reflect real transactions.

The primary source of uncertainty in this pipeline is the anonymization of features: because `V1`–`V28` have no interpretable meaning, it is impossible to audit whether the model is relying on legitimate behavioral signals or artifacts of the PCA transformation.

---

## Metadata

### Implicit Schema
Each document in the `transactions` collection follows this structure:

```json
{
  "_id": "ObjectId",
  "Time": "float",
  "V1": "float",
  "...": "...",
  "V28": "float",
  "Amount": "float",
  "Class": "int"
}
```

All documents contain all 31 fields. `Class` is always `0` or `1`. `Amount` is always non-negative. `Time` is always non-negative and monotonically increasing within the original dataset. `V1`--`V28` are continuous floats with no guaranteed range (PCA-derived). No nested arrays or subdocuments are used; every field is a scalar.

### Data Summary

| Property | Value |
|:---------|:------|
| Collection | `transactions` |
| Total documents | 284,807 |
| Fraudulent (`Class = 1`) | 492 (0.172%) |
| Legitimate (`Class = 0`) | 284,315 (99.828%) |
| Fields per document | 31 |
| `Time` range | 0 to 172,792 seconds |
| `Amount` range | $0.00 to $25,691.16 |
 

### Data Dictionary

| Feature | Type | Description | Example |
|:--------|:-----|:------------|:--------|
| `_id` | ObjectId | MongoDB-assigned document identifier | `ObjectId("...")` |
| `Time` | float | Seconds elapsed between this transaction and the first transaction in the dataset | `0.0`, `406.0` |
| `V1`--`V28` | float | Anonymized PCA components; original features withheld for confidentiality | `1.192`, `-0.966` |
| `Amount` | float | Transaction amount in euros | `149.62`, `2.69` |
| `Class` | int | Fraud label: `1` = fraudulent, `0` = legitimate | `0`, `1` |


### Uncertainty Quantification

| Feature | Mean | Std Dev | Min | Max | Notes |
|:--------|:-----|:--------|:----|:----|:------|
| `Time` | 94,814 | 47,488 | 0 | 172,792 | Roughly uniform across 48 hours; no missing values |
| `Amount` | 88.35 | 250.12 | 0.00 | 25,691.16 | Highly right-skewed; large outliers from rare high-value transactions; should be scaled before modeling |
| `V1`--`V28` | ~0 | 0.33 to 1.96 | varies | varies | Means are effectively zero by PCA construction; standard deviations vary by component and are not unit variance; no interpretable meaning |
| `Class` | 0.0017 | 0.041 | 0 | 1 | Binary; uncertainty stems entirely from class imbalance, not measurement error |