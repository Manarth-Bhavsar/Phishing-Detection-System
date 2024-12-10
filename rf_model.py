import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from db import LogEntry  # Database model for logs
import pickle
from Feature_ext import generate_dataset  # Feature extraction logic

# Function to prepare data for training and testing
def prepare_data(data):
    """
    Splits the dataset into features (X) and target (y), 
    and further into training and testing sets.
    """
    y = data['Result']
    X = data.drop('Result', axis=1)  # Drop the target column to get features
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

def train_and_save_model(data, model_path='rfmodel.pkl'):
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(data)
    
    # Train the model
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    rf_acc = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {rf_acc * 100:.2f}%")
    
    # Save the model
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_path}")

def load_model(model_path='rfmodel.pkl'):
    try:
        with open(model_path, 'rb') as model_file:
            model = pickle.load(model_file)
        print(f"Model loaded successfully from {model_path}")
        return model
    except FileNotFoundError:
        print(f"Model file not found at {model_path}. Ensure the file path is correct.")
        raise
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        raise


# Function to retrain the model using data from the database
def retrain_model():
    """
    Retrains the model using data stored in the database logs.
    Extracts features from stored URLs, retrains the model, and saves it.
    """
    # Fetch all logs from the database
    logs = LogEntry.query.all()
    
    features_list = []
    labels = []
    
    for log in logs:
        try:
            # Extract features for the URL
            features, _ = generate_dataset(log.url)
            
            # Append features and corresponding label
            features_list.append(features)
            labels.append(1 if log.is_phishing else -1)  # Adjust label mapping as per the database
        except Exception as e:
            print(f"Error processing URL {log.url}: {e}")
            continue

    # Ensure we have data to retrain
    if not features_list or not labels:
        raise ValueError("No valid data available for retraining")

    # Define the feature columns (ensure consistency with initial training)
    feature_columns = [
        'having_IP_Address', 'URL_Length', 'Shortining_Service', 'having_At_Symbol',
        'double_slash_redirecting', 'Prefix_Suffix', 'having_Sub_Domain', 'SSLfinal_State',
        'Domain_registeration_length', 'Favicon', 'port', 'HTTPS_token', 'Request_URL',
        'URL_of_Anchor', 'Links_in_tags', 'SFH', 'Submitting_to_email', 'Abnormal_URL',
        'Redirect', 'on_mouseover', 'RightClick', 'popUpWidnow', 'Iframe', 'age_of_domain',
        'DNSRecord', 'web_traffic', 'Page_Rank', 'Google_Index', 'Links_pointing_to_page',
        'Statistical_report'
    ]

    # Create the dataset for retraining
    X = pd.DataFrame(features_list, columns=feature_columns)
    y = pd.Series(labels)

    # Train a new Random Forest model
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)

    # Save the retrained model
    with open('rfmodel.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print("Model retrained successfully and saved to 'rfmodel.pkl'")
    return "Model retrained successfully!"


if __name__ == '__main__':
    # Load dataset
    data = pd.read_csv('PhisingWebsite_datset.csv')
    
    # Train and save the model
    train_and_save_model(data)
