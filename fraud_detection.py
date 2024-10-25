import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from scipy import stats
import logging  
# from kafka import KafkaConsumer
from datetime import datetime
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "PS_20174392719_1491204439457_log.csv")

timestamp = datetime.now().strftime("%Y%m%d_%H")
log_filename = os.path.join(script_dir, f"sahtekarlik_analiz_log_{timestamp}.txt")

logging.basicConfig(
    filename=log_filename, 
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


#Bu kısımı kafka için ekledim ama şuan çalışmadığı için yorum satırı olarak düzelttim. Araştırma yapıyorum.

# consumer = KafkaConsumer(
#     'fraud_topic',  
#     bootstrap_servers='localhost:9092',  
#     value_deserializer=lambda x: json.loads(x.decode('utf-8'))
# )

# data_list = []
# for message in consumer:
#     data_list.append(message.value)  
#     if len(data_list) >= 100: 
#         break

# data = pd.DataFrame(data_list)


data = pd.read_csv(data_path)
print(data.head())
print(data.isnull().sum())


data.fillna(0, inplace=True)
logging.info("Eksik veriler temizlendi.")


data = pd.get_dummies(data, columns=['type'])
logging.info("Kategorik veriler kodlandi.")


upper_limit = data['amount'].quantile(0.99)
lower_limit = data['amount'].quantile(0.01)
data_clean = data[(data['amount'] < upper_limit) & (data['amount'] > lower_limit)].copy()
logging.info("Aykiri degerler temizlendi.")


X = data_clean.drop(['isFraud', 'nameOrig', 'nameDest'], axis=1)
y = data_clean['isFraud']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
logging.info("Veri olceklendi.")


smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
logging.info("SMOTE ile sinif dengesizligi giderildi.")


X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
logging.info("Veri egitim ve test setlerine bolundu.")


model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
logging.info("Model egitildi.")


y_pred = model.predict(X_test)


conf_matrix = confusion_matrix(y_test, y_pred)
logging.info("Karmaşa matrisi:\n%s", conf_matrix)

class_report = classification_report(y_test, y_pred)
logging.info("Siniflandirma raporu:\n%s", class_report)


fraud_rate = (data_clean['isFraud'].sum() / len(data_clean)) * 100
total_transactions = len(data_clean)
total_frauds = data_clean['isFraud'].sum()

logging.info(f"Toplam Işlem Sayisi: {total_transactions}")
logging.info(f"Sahtekarlik Olay Sayisi: {total_frauds}")
logging.info(f"Sahtekarlik Orani: {fraud_rate:.2f}%")


z_scores = np.abs(stats.zscore(data_clean.select_dtypes(include=[np.number])).values)


plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
sns.histplot(z_scores.flatten(), bins=50, color='blue')
plt.title('Z-Skoru Histogrami')

plt.subplot(1, 2, 2)
sns.boxplot(x=data_clean['amount'])
plt.title('Amount Boxplot')

plt.savefig(os.path.join(script_dir, "sahtekarlik_analiz_grafikleri.png"))
logging.info("Grafikler kaydedildi.")

print("İşlemler ve grafik loglama tamamlandı.")