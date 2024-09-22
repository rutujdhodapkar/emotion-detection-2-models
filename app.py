import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load the dataset and cache it
@st.cache_data
def load_data(data_path):
    return pd.read_csv(data_path)

# PyTorch Neural Network Model
class EmotionModel(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size):
        super(EmotionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.fc4 = nn.Linear(hidden_size3, output_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Vectorization and model training for Logistic Regression, cached to speed up execution
@st.cache_resource
def train_logistic_regression(X_train, y_train):
    vectorizer = TfidfVectorizer(max_features=8000)
    X_train_vec = vectorizer.fit_transform(X_train)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)
    return vectorizer, model

# Function to load PyTorch model (cached to avoid reloading every time)
@st.cache_resource
def load_pytorch_model(input_size, output_size):
    model = EmotionModel(input_size=input_size, hidden_size1=2048, hidden_size2=1024, hidden_size3=512, output_size=output_size)
    model.load_state_dict(torch.load('emotion_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

# Prediction function for PyTorch model
def predict_emotion_pytorch(text, model, vectorizer, label_mapping):
    text_vec = vectorizer.transform([text]).toarray()
    text_tensor = torch.tensor(text_vec, dtype=torch.float32)
    with torch.no_grad():
        output = model(text_tensor)
        _, predicted = torch.max(output, 1)
        emotion = list(label_mapping.keys())[list(label_mapping.values()).index(predicted.item())]
        return emotion

# Prediction function for Logistic Regression
def predict_emotion_logistic(text, vectorizer, model):
    text_vec = vectorizer.transform([text])
    return model.predict(text_vec)[0]

# Load the dataset
data_path = 'Emotion_final_with_predictions.csv'  # Ensure the path is correct for deployment
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

# Initialize Streamlit app
st.title('Emotion Prediction with PyTorch and Logistic Regression')

# Let the user choose the model
model_choice = st.selectbox("Choose a model for emotion prediction", ("Logistic Regression", "PyTorch Neural Network"))

# Input text for prediction
input_text = st.text_input("Enter text for emotion prediction:")

if st.button('Predict'):
    if input_text:
        if model_choice == "Logistic Regression":
            predicted_emotion = predict_emotion_logistic(input_text, vectorizer_logistic, logistic_regression_model)
            st.write(f'Predicted Emotion (Logistic Regression): {predicted_emotion}')
        else:
            # Load PyTorch model (assuming it is pre-trained and saved as a .pth file)
            vectorizer_pytorch = TfidfVectorizer(max_features=8000)
            vectorizer_pytorch.fit(X_train)  # Same vectorizer used for training
            pytorch_model = load_pytorch_model(input_size=8000, output_size=len(label_mapping))
            predicted_emotion = predict_emotion_pytorch(input_text, pytorch_model, vectorizer_pytorch, label_mapping)
            st.write(f'Predicted Emotion (PyTorch Neural Network): {predicted_emotion}')
    else:
        st.write("Please enter some text to predict the emotion.")

# Display the dataset and evaluation metrics for Logistic Regression
st.subheader('Dataset')
st.write(data.head())

# Evaluate Logistic Regression model
X_test_vec_logistic = vectorizer_logistic.transform(X_test)
y_pred_logistic = logistic_regression_model.predict(X_test_vec_logistic)
accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
st.write(f'Logistic Regression Accuracy: {accuracy_logistic:.4f}')
st.text(classification_report(y_test, y_pred_logistic))
