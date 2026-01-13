import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference

# Sample data for tests
@pytest.fixture
def sample_data():
    data = pd.DataFrame({
        "age": [25, 38, 28, 44],
        "workclass": ["Private", "Self-emp-not-inc", "Private", "Private"],
        "education": ["Bachelors", "HS-grad", "Masters", "Some-college"],
        "marital-status": ["Never-married", "Married-civ-spouse", "Divorced", "Married-civ-spouse"],
        "occupation": ["Tech-support", "Exec-managerial", "Prof-specialty", "Sales"],
        "relationship": ["Not-in-family", "Husband", "Own-child", "Husband"],
        "race": ["White", "Black", "White", "Black"],
        "sex": ["Male", "Male", "Female", "Male"],
        "native-country": ["United-States", "United-States", "United-States", "United-States"],
        "salary": ["<=50K", ">50K", "<=50K", ">50K"]
    })
    cat_features = [
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country"
    ]
    return data, cat_features

# ------------------------
# Test 1: train_model returns a RandomForestClassifier
# ------------------------
def test_train_model_type(sample_data):
    data, cat_features = sample_data
    X, y, encoder, lb = process_data(data, categorical_features=cat_features, label="salary", training=True)
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier), "train_model should return a RandomForestClassifier"

# ------------------------
# Test 2: compute_model_metrics returns numbers between 0 and 1
# ------------------------
def test_compute_model_metrics_values(sample_data):
    data, cat_features = sample_data
    X, y, encoder, lb = process_data(data, categorical_features=cat_features, label="salary", training=True)
    model = train_model(X, y)
    preds = model.predict(X)
    precision, recall, fbeta = compute_model_metrics(y, preds)
    for metric in [precision, recall, fbeta]:
        assert 0 <= metric <= 1, "Metrics should be between 0 and 1"

# ------------------------
# Test 3: process_data returns arrays of expected shape
# ------------------------
def test_process_data_shapes(sample_data):
    data, cat_features = sample_data
    X, y, encoder, lb = process_data(data, categorical_features=cat_features, label="salary", training=True)
    assert isinstance(X, np.ndarray), "X should be a numpy array"
    assert isinstance(y, np.ndarray), "y should be a numpy array"
    assert X.shape[0] == y.shape[0], "X and y should have the same number of rows"
