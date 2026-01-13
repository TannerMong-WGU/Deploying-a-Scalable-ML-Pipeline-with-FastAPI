# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Model Type:
Random Forest Classifier

Implementation:
sklearn.ensemble.RandomForestClassifier

Hyperparameters:
n_estimators=100
max_depth=None
random_state=42

Categorical features:
workclass
education
marital-status
occupation
relationship
race
sex
native-country

Label: 
salary (>50K or <=50K)

Preprocessing:
One-hot encoding for categorical features, label binarization for the target variable.

## Intended Use
Primary purpose:
Predict whether an individual’s income exceeds 50K/year based on census data features.

Intended users:
Researchers, analysts, or applications that require income prediction for analysis or modeling purposes.

Not intended for:
Any high-stakes decision-making (e.g., lending, hiring, insurance) without additional fairness and ethical review.

## Training Data
Dataset:
Census income dataset (census.csv)

Train/Test split:
80/20

Number of training samples:
~29,000 (exact depends on dataset)

Number of features:
8 categorical + 6 numerical (processed with process_data)

## Evaluation Data
Test set:
20% of the dataset (~7,500 samples)

Processing:
Same as training (one-hot encoding using training encoders)

## Metrics
Overall performance on test set:
Precision:
0.7419

Recall:
0.6384

F1-score:
0.6863

Performance on categorical slices:
Can be found in the slice_output.txt file in this repo.

## Ethical Considerations
Bias:
Certain slices (e.g., small educational groups, rare workclasses) show lower recall or precision, which may disproportionately affect underrepresented groups.

Privacy:
Uses publicly available census data; no individual-identifiable data is exposed.

Intended use caution:
Model predictions should not be used for high-stakes decisions without further fairness auditing.

## Caveats and Recommendations
Model may underperform for rare categories (e.g., very small workclasses or countries).

Metrics like F1 vary widely across slices; users should inspect slice performance before deployment.

Consider retraining with more data or different sampling techniques to mitigate disparities.

Always pair predictions with ethical and legal considerations before real-world use.