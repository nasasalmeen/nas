import pickle
import os
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
application = app  # Required for Elastic Beanstalk

MODEL_PATH = 'model.pkl'

def load_titanic_data():
    """Load Titanic dataset from a URL."""
    url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
    df = pd.read_csv(url)
    return df

def preprocess_data(df):
    """Preprocess the Titanic dataset."""
    # Select features
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

    # Create a copy with only needed columns
    data = df[features + ['Survived']].copy()

    # Handle missing values
    data['Age'] = data['Age'].fillna(data['Age'].median())
    data['Fare'] = data['Fare'].fillna(data['Fare'].median())
    data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])

    # Encode categorical variables
    le_sex = LabelEncoder()
    le_embarked = LabelEncoder()

    data['Sex'] = le_sex.fit_transform(data['Sex'])
    data['Embarked'] = le_embarked.fit_transform(data['Embarked'])

    X = data[features]
    y = data['Survived']

    return X, y, le_sex, le_embarked

def train_model():
    """Train and save the model."""
    print("Loading Titanic dataset...")
    df = load_titanic_data()

    print("Preprocessing data...")
    X, y, le_sex, le_embarked = preprocess_data(df)

    print("Training Random Forest model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"Model accuracy: {accuracy:.2f}")

    # Save model and encoders
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump({
            'model': model,
            'le_sex': le_sex,
            'le_embarked': le_embarked
        }, f)

    print(f"Model saved to {MODEL_PATH}")
    return model, le_sex, le_embarked

def load_model():
    """Load the trained model."""
    if not os.path.exists(MODEL_PATH):
        return train_model()

    with open(MODEL_PATH, 'rb') as f:
        data = pickle.load(f)
    return data['model'], data['le_sex'], data['le_embarked']

# Load model at startup
model, le_sex, le_embarked = load_model()

@app.route('/')
def index():
    """Display the prediction form."""
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    """Make a prediction based on form input."""
    try:
        # Get form data
        pclass = int(request.form['pclass'])
        sex = request.form['sex']
        age = float(request.form['age'])
        sibsp = int(request.form['sibsp'])
        parch = int(request.form['parch'])
        fare = float(request.form['fare'])
        embarked = request.form['embarked']

        # Encode categorical variables
        sex_encoded = le_sex.transform([sex])[0]
        embarked_encoded = le_embarked.transform([embarked])[0]

        # Create feature array
        features = np.array([[pclass, sex_encoded, age, sibsp, parch, fare, embarked_encoded]])

        # Make prediction
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]

        result = {
            'survived': bool(prediction),
            'probability': f"{probability[1] * 100:.1f}%",
            'probability_value': round(probability[1] * 100, 1)
        }

        return render_template('index.html', prediction=result, form_data=request.form)

    except Exception as e:
        return render_template('index.html', prediction=None, error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
