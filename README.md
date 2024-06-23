

# Language Detection Model

This project is a language detection model that uses a Naive Bayes classifier to predict the language of a given text. The model is trained on a dataset of text samples labeled with their respective languages.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Prediction Function](#prediction-function)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. **Clone the repository**:
   ```
   git clone https://github.com/yourusername/language-detection.git
   cd language-detection
   ```

2. **Install the required libraries**:
   ```
   pip install pandas numpy seaborn matplotlib scikit-learn
   ```

## Usage

1. **Load the dataset**:
   Ensure you have the dataset file (e.g., `language_dataset.csv`) in the project directory.

2. **Run the script**:
   ```
   python language_detection.py
   ```

3. **Predict the language**:
   Use the `predict` function to predict the language of a given text. Examples are provided in the script.

## Project Structure

- `language_detection.py`: Main script containing the code for data preprocessing, model training, evaluation, and prediction.
- `language_dataset.csv`: Dataset file containing text samples and their respective languages.

## Model Training and Evaluation

1. **Data Loading and Preprocessing**:
   - Load the dataset using Pandas.
   - Separate the independent (`Text`) and dependent (`Language`) features.
   - Encode the categorical `Language` feature using `LabelEncoder`.
   - Preprocess the text data by removing symbols and numbers, converting to lowercase, and creating a bag-of-words representation using `CountVectorizer`.

2. **Train-Test Split**:
   - Split the data into training and testing sets using `train_test_split`.

3. **Model Training**:
   - Train a Naive Bayes classifier (`MultinomialNB`) on the training data.

4. **Model Evaluation**:
   - Predict the labels for the test data.
   - Evaluate the model using accuracy score and confusion matrix.
   - Visualize the confusion matrix using Seaborn's heatmap.

## Prediction Function

The `predict` function takes a text input, preprocesses it, and predicts the language using the trained model. Examples of predictions for different languages are provided in the script.

```
def predict(text):
    x = cv.transform([text]).toarray()
    lang = model.predict(x)
    lang = le.inverse_transform(lang)
    print("The language is", lang[0])
```

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.
