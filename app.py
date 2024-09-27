import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Check if GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available, else CPU

# Load the dataset
@st.cache_data
def load_data(data_path):
    return pd.read_csv(data_path)

# PyTorch Neural Network Model
class EmotionModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(EmotionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 1000)   # Adjust as needed for parameter count
        self.fc2 = nn.Linear(1000, 500)          # Adjust as needed for parameter count
        self.fc3 = nn.Linear(500, output_size)    # Output layer
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Function to calculate total parameters in the PyTorch model
def calculate_total_params(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params

# Vectorization and model training for Logistic Regression
def train_logistic_regression(X_train, y_train):
    vectorizer = TfidfVectorizer(max_features=8000)
    X_train_vec = vectorizer.fit_transform(X_train)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)
    return vectorizer, model

# Function to load PyTorch model
def load_pytorch_model(input_size, output_size):
    model = EmotionModel(input_size=input_size, output_size=output_size)
    model.to(device)  # Move the model to GPU or CPU
    return model

# Prediction function for PyTorch model
def predict_emotion_pytorch(text, model, vectorizer, label_mapping):
    text_vec = vectorizer.transform([text]).toarray()
    text_tensor = torch.tensor(text_vec, dtype=torch.float32).to(device)  # Move tensor to GPU or CPU
    with torch.no_grad():
        output = model(text_tensor)
        _, predicted = torch.max(output, 1)
        emotion = list(label_mapping.keys())[list(label_mapping.values()).index(predicted.item())]
        return emotion

# Prediction function for Logistic Regression
def predict_emotion_logistic(text, vectorizer, model):
    text_vec = vectorizer.transform([text])
    return model.predict(text_vec)[0]

# Load the dataset and prepare data
data_path = 'Emotion_final_with_predictions.csv'  # Ensure this file is in the same directory
data = load_data(data_path)

# Prepare data
X = data['Text']
y = data['Emotion']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Map emotions to numerical labels for PyTorch
label_mapping = {emotion: idx for idx, emotion in enumerate(y.unique())}
y_train_mapped = y_train.map(label_mapping)
y_test_mapped = y_test.map(label_mapping)

# Train logistic regression model
vectorizer_logistic, logistic_regression_model = train_logistic_regression(X_train, y_train)

# Load PyTorch model
vectorizer_pytorch = TfidfVectorizer(max_features=8000)
vectorizer_pytorch.fit(X_train)  # Same vectorizer used for training
pytorch_model = load_pytorch_model(input_size=8000, output_size=len(label_mapping))

# Calculate total parameters in the PyTorch model
total_params = calculate_total_params(pytorch_model)

# Streamlit UI setup
st.title("Emotion Prediction App")
st.write("Enter a sentence below to predict the emotion:")

# Display the total number of parameters
st.write(f'**Total Parameters in the PyTorch Model:** {total_params}')

# Text input for user
user_input = st.text_area("Input Text:")

if st.button("Predict Emotion"):
    if user_input:
        predicted_emotion_pytorch = predict_emotion_pytorch(user_input, pytorch_model, vectorizer_pytorch, label_mapping)
        predicted_emotion_logistic = predict_emotion_logistic(user_input, vectorizer_logistic, logistic_regression_model)

        st.write(f'**Predicted Emotion (PyTorch Neural Network):** {predicted_emotion_pytorch}')
        st.write(f'**Predicted Emotion (Logistic Regression):** {predicted_emotion_logistic}')
    else:
        st.write("Please enter some text for prediction.")

# Evaluate Logistic Regression model (optional, could be run separately)
X_test_vec_logistic = vectorizer_logistic.transform(X_test)
y_pred_logistic = logistic_regression_model.predict(X_test_vec_logistic)
accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
st.write(f'**Logistic Regression Accuracy on Test Set:** {accuracy_logistic:.4f}')
