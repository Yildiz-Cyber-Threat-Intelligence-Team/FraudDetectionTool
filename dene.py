import argparse
import pandas as pd
import numpy as np
import logging
import json
from datetime import datetime, time
import random
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
import zmq
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from numpy.random import choice
from pathlib import Path

timestamp = datetime.now().strftime("%Y%m%d_%H")
script_dir = Path(__file__).resolve().parent
logfile = script_dir / 'fraud_detection.log'
file_path = script_dir / 'fraud_data.csv'
model_path = script_dir / 'fraud_model.pkl'
graphic_path = script_dir / 'fraud_analysis_graphic.png'

logging.basicConfig(
    filename=f"{logfile}{timestamp}.log",
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Fraud detection and analysis tool with ZeroMQ and ML integration.')
    parser.add_argument('--file', type=str, required=True, help='CSV file path for initial data loading')
    parser.add_argument('--operation', type=str, choices=[
        'head', 'describe', 'shape', 'fraud_rate', 'train_model',
        'stream_zeromq', 'zeromq_consumer'
    ], required=True, help='Operation to perform')
    parser.add_argument('--output', type=str, default='zeromq_output.json', help='Output file for ZeroMQ results')
    return parser.parse_args()

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        data.fillna(data.median(axis=0), inplace=True)
        data.ffill(inplace=True)
        Q1 = data['amount'].quantile(0.25)
        Q3 = data['amount'].quantile(0.75)
        IQR = Q3 - Q1
        lower_limit = Q1 - 1.5 * IQR
        upper_limit = Q3 + 1.5 * IQR
        data = data[(data['amount'] < upper_limit) & (data['amount'] > lower_limit)]
        return data
    except Exception as e:
        logging.error(f"Data loading error: {e}")
        return None

def preprocess_data(data):
    numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    required_cols = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'isFraud']
    for col in required_cols:
        if col not in data.columns:
            raise ValueError(f"Missing required column: {col}")
    data_clean = data[required_cols].copy()
    Q1 = data['amount'].quantile(0.25)
    Q3 = data['amount'].quantile(0.75)
    IQR = Q3 - Q1
    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR
    data_clean = data_clean[(data_clean['amount'] < upper_limit) & (data_clean['amount'] > lower_limit)]
    X = data_clean.drop(['isFraud'], axis=1)
    y = data_clean['isFraud']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, data_clean

def train_model(X_train, y_train, model_path):
    model = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42
    )
    model.fit(X_train, y_train)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    logging.info(f"Model {model_path} yoluna kaydedildi.")
    return model

def evaluate_model(model, X_test, y_test, data_clean):
    y_pred = model.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    logging.info(f"Confusion Matrix:\n{conf_matrix}")
    logging.info(f"Classification Report:\n{class_report}")
    for i in range(len(y_test)):
        if y_test.iloc[i] != y_pred[i]:
            logging.info(f"Hatalı Tahmin: Gerçek Değer: {y_test.iloc[i]}, Tahmin: {y_pred[i]}")
            logging.info(f"İşlem Bilgileri: {data_clean.iloc[i]}")

def plot_data_analysis(data_clean):
    z_scores = np.abs(stats.zscore(data_clean.select_dtypes(include=[np.number])).values)
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(z_scores.flatten(), bins=50, color='blue')
    plt.title('Z-Skoru Histogramı')
    plt.subplot(1, 2, 2)
    sns.boxplot(x=data_clean['amount'])
    plt.title('Amount Boxplot')
    plt.savefig(f'{graphic_path}{timestamp}.png')
    logging.info("Grafikler kaydedildi.")

def stream_zeromq():
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind("tcp://*:5555")
    while True:
        transaction = {
            "transaction_id": random.randint(10000, 99999),
            "amount": round(random.uniform(1.0, 1000.0), 2),
            "timestamp": timestamp,
            "is_fraud": choice([0, 1], p=[0.8, 0.2]),
            "location": random.choice(['US', 'UK', 'DE', 'FR']),
            "payment_method": random.choice(['CREDIT_CARD', 'PAYPAL', 'BANK_TRANSFER']),
            "oldbalanceOrg": round(random.uniform(0.0, 10000.0), 2),
            "newbalanceOrig": round(random.uniform(0.0, 10000.0), 2),
            "oldbalanceDest": round(random.uniform(0.0, 10000.0), 2)
        }
        # Tüm int32 türündeki değerleri int türüne dönüştür
        transaction = {key: int(value) if isinstance(value, np.integer) else value for key, value in transaction.items()}
        socket.send_json(transaction)
        logging.info(f"Sent transaction: {transaction}")
        time.sleep(2)

def zeromq_consumer(output_file='zeromq_output.json', model_path='fraud_model.pkl'):
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    except Exception as e:
        logging.error(f"Model loading error: {e}")
        return
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect("tcp://localhost:5555")
    socket.setsockopt_string(zmq.SUBSCRIBE, '')
    with open(output_file, 'a') as f:
        while True:
            try:
                transaction = socket.recv_json()
                features = [
                    int(transaction['amount']),
                    int(transaction['oldbalanceOrg']),
                    int(transaction['newbalanceOrig']),
                    int(transaction['oldbalanceDest'])
                ]
                prediction = model.predict([features])
                transaction['prediction'] = int(prediction[0])
                f.write(json.dumps(transaction) + '\n')
                logging.info(f"Transaction Processed: {transaction}")
            except Exception as e:
                logging.error(f"Error processing transaction: {str(e)}")
                logging.error(f"Failed transaction data: {transaction}")

def process_data(data, operation, model_path, output_file=None):
    if operation == 'train_model':
        X_train, X_test, y_train, y_test, data_clean = preprocess_data(data)
        model = train_model(X_train, y_train, model_path)
        evaluate_model(model, X_test, y_test, data_clean)
        plot_data_analysis(data_clean)
    elif operation == 'stream_zeromq':
        stream_zeromq()
    elif operation == 'zeromq_consumer':
        zeromq_consumer(output_file=output_file, model_path=model_path)
    else:
        if operation == 'head':
            logging.info(f"First 5 rows:\n{data.head()}")
        elif operation == 'describe':
            logging.info(f"Data description:\n{data.describe()}")
        elif operation == 'shape':
            logging.info(f"Data shape: {data.shape}")
        elif operation == 'fraud_rate':
            fraud_rate = (data['isFraud'].sum() / len(data)) * 100
            logging.info(f"Fraud rate: {fraud_rate:.2f}%")

if __name__ == "__main__":
    args = parse_arguments()
    data = load_data(args.file)
    if data is not None:
        process_data(data, args.operation, output_file=args.output)
