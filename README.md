# Iris Classification Model

A basic supervised learning project that trains a Decision Tree classifier
on the classic Iris flower dataset — covering the full ML workflow from
raw data to an evaluated, working model.

## Features

- Loads and explores the Iris dataset (150 samples, 4 features, 3 classes)
- Splits data into stratified training (80%) and testing (20%) sets
- Trains a Decision Tree Classifier
- Evaluates with accuracy, confusion matrix, and a classification report
- Shows feature importances (which measurements matter most)
- Predicts the species of a new, unseen flower sample

## Results

| Metric | Score |
|---|---|
| Accuracy | 96.67% |

```
              precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        10
  versicolor       1.00      0.90      0.95        10
   virginica       0.91      1.00      0.95        10

    accuracy                           0.97        30
   macro avg       0.97      0.97      0.97        30
weighted avg       0.97      0.97      0.97        30
```

## Dataset

The [Iris dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#iris-plants-dataset)
is a classic beginner dataset with 150 samples of iris flowers across 3
species (setosa, versicolor, virginica), described by 4 features:
sepal length, sepal width, petal length, and petal width.

## Getting Started

### Prerequisites
- Python 3.7+
- pandas
- scikit-learn

### Installation

```bash
git clone https://github.com/<your-username>/iris-classification-model.git
cd iris-classification-model
pip install -r requirements.txt
```

### Run it

```bash
python classification_model.py
```

## Project Workflow

1. **Load and understand the dataset** — inspect shape, class balance, and summary statistics
2. **Split the data** — training and testing sets using `train_test_split`
3. **Train the model** — a simple Decision Tree Classifier
4. **Evaluate** — accuracy, confusion matrix, precision/recall/F1
5. **Predict** — classify a brand-new, unseen sample

## Future Improvements

- Compare multiple algorithms (Logistic Regression, KNN, SVM)
- Add cross-validation
- Visualize the decision tree and decision boundaries
- Hyperparameter tuning with GridSearchCV

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Author

**Manan** — BS Artificial Intelligence student
