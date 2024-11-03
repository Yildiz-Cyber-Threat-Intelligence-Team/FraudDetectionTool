# FraudDetectionTool

# TR

Bu proje, fraud detection(sahtecilik tespiti) için yapılmış bir araçtır. Makine öğrenimi kullanarak çalışır. ZeroMQ ile anlık veri akışı sağlar, büyük miktardaki işlemler için Telegram'dan uyarı gönderir ve ayrıca farklı analizler de yapabilir. Birçok parametre ile esnek bir analiz ve veri işleme imkanı sunar.

## Özellikler
- Fraud tespiti için bir makine öğrenimi modeli oluşturur, eğitir ve test eder.
- ZeroMQ kullanarak sahte işlemleri anlık olarak yayınlar.
- Veriyi analiz etmek için grafikler oluşturur ve sonuçları, uyarıları log dosyalarına kaydeder.
- Yüksek miktarlı veya şüpheli işlemler tespit edildiğinde Telegram üzerinden uyarılar gönderir.

## Kullanımı
```python fraud_detection.py --file <VERI_DOSYASI_YOLU> --operation <İŞLEM> --output <ÇIKTI_DOSYASI> --bot_token <BOT_TOKEN> --chat_id <CHAT_ID>```

## Parametreler
`--file` **(gerekli)**: İşlenecek CSV dosyasının yolunu belirtir. Örneğin, `data.csv`.
`--operation` **(gerekli)**: Gerçekleştirilecek işlemi belirler. Aşağıdaki işlemlerden biri olabilir:
- `train_model`: Model eğitimi ve değerlendirme işlemini başlatır.
- `fraud_rate`: Sahtecilik oranını hesaplar.
- `stream_zeromq`: Gerçek zamanlı veri akışını başlatır.
- `compare`: Gerçek ve sahte işlemler arasındaki farkları karşılaştırır.
- `plot_fraud`: Sahte ve gerçek işlemlerin dağılımını grafikle gösterir.
- `fraud_categories`: En çok sahte işlemler yapılan işlem türlerini listeler.
- `fraud_trend`: Aylık sahtecilik trendlerini gösterir.
- `monthly_fraud_count`: Aylık sahte işlem sayısını listeler.
- `fraud_rate_by_type`: İşlem türlerine göre sahtecilik oranını hesaplar.
- `correlation_heatmap`: Korelasyon ısı haritasını gösterir.
- `high_value_alerts`: Yüksek tutarlı işlemleri belirler ve log kaydına ekler.
`--output`: ZeroMQ tüketici çıktısının kaydedileceği dosya. Yalnızca `zeromq_consumer` işlemi için kullanılır.
`--bot_token`: Telegram bot token’ı. `zeromq_consumer` ile Telegram uyarıları göndermek için kullanılır.
`--chat_id`: Telegram chat ID’si. `zeromq_consumer` ile Telegram üzerinden uyarı göndermek için kullanılır.


## Fonksiyonlar

### 1. Veri Yükleme ve Hazırlık

- `load_data`: Veriyi dosyadan yükler, eksik veya uç değerleri temizler.
- `preprocess_data`: Veriyi analiz için hazırlar ve sahte işlem sınıflarını dengelemek için SMOTE tekniğini uygular.

### 2. Model Eğitimi ve Değerlendirme

- `train_model`: Veriyi kullanarak modeli eğitir ve kaydeder.
- `evaluate_model`: Modelin doğruluğunu test eder ve değerlendirme raporlarını oluşturur.

### 3. Grafik ve Görselleştirme

- `plot_data_analysis`: Veriyi analiz etmek için histogram ve boxplot grafikleri oluşturur, dosyaya kaydeder.

### 4. Gerçek Zamanlı Veri Akışı ve Tüketim

- `stream_zeromq`: Rastgele sahte işlem verilerini gerçek zamanlı olarak yayınlar.
- `zeromq_consumer`: Yayınlanan verileri alır, modelle değerlendirir ve Telegram ile uyarılar gönderir.

### 5. Telegram Uyarı Sistemi

- `send_telegram_alert`: Şüpheli işlem tespit edildiğinde işlem bilgilerini Telegram’a gönderir.

### 6. Ek Analiz İşlevleri

- `compare_fraud_nonfraud`: Sahte ve gerçek işlemleri karşılaştırır.
- `plot_fraud_distribution`: Sahte ve gerçek işlem sayılarının dağılımını gösterir.
- `fraud_categories`: En çok sahte işlemler yapılan işlem türlerini listeler.
- `fraud_trend`: Aylık sahtecilik trendlerini analiz eder.
- `monthly_fraud_count`: Aylık sahte işlem sayılarını listeler.
- `fraud_rate_by_type`: İşlem türlerine göre sahtecilik oranını hesaplar.
- `correlation_heatmap`: Verideki özellikler arasındaki ilişkileri gösteren bir ısı haritası oluşturur.
- `high_value_alerts`: Yüksek miktarlı işlemleri belirler ve log kaydına ekler.



# EN
This project is a tool created for fraud detection. It works using machine learning. It provides real-time data streaming with ZeroMQ, sends alerts via Telegram for high-value transactions, and can also perform various analyses. It offers flexible options for analysis and data processing with many parameters.

## Features

- Creates, trains, and tests a machine learning model for fraud detection.
- Uses ZeroMQ to stream fraudulent transactions in real-time.
- Creates charts to analyze the data and logs results and alerts.
- Sends alerts via Telegram when high-value or suspicious transactions are detected.

## Usage
`python fraud_detection.py --file <DATA_FILE_PATH> --operation <OPERATION> --output <OUTPUT_FILE> --bot_token <BOT_TOKEN> --chat_id <CHAT_ID>`

## Parameters

- `--file` **(required)**: Specifies the path to the CSV file to be processed. For example, `data.csv`.
- `--operation` **(required)**: Defines the operation to be performed. It can be one of the following:
    - `train_model`: Initiates model training and evaluation.
    - `fraud_rate`: Calculates the fraud rate.
    - `stream_zeromq`: Starts real-time data streaming.
    - `compare`: Compares the differences between fraudulent and genuine transactions.
    - `plot_fraud`: Shows the distribution of fraudulent and genuine transactions in a chart.
    - `fraud_categories`: Lists the transaction types with the most fraud.
    - `fraud_trend`: Shows monthly fraud trends.
    - `monthly_fraud_count`: Lists the monthly fraud count.
    - `fraud_rate_by_type`: Calculates the fraud rate by transaction type.
    - `correlation_heatmap`: Displays a correlation heatmap.
    - `high_value_alerts`: Identifies high-value transactions and logs them.
- `--output`: The file to save the ZeroMQ consumer output. Only used for the `zeromq_consumer` operation.
- `--bot_token`: Telegram bot token. Used to send alerts via Telegram with the `zeromq_consumer` operation.
- `--chat_id`: Telegram chat ID. Used to send alerts via Telegram with the `zeromq_consumer` operation.


## Functions

### 1. Data Loading and Preparation

- `load_data`: Loads the data from a file and cleans missing or outlier values.
- `preprocess_data`: Prepares the data for analysis and applies the SMOTE technique to balance fraud and non-fraud classes.

### 2. Model Training and Evaluation

- `train_model`: Trains the model using the data and saves it.
- `evaluate_model`: Tests the model's accuracy and generates evaluation reports.

### 3. Charting and Visualization

- `plot_data_analysis`: Creates histogram and boxplot charts for data analysis and saves them.

### 4. Real-Time Data Streaming and Consumption

- `stream_zeromq`: Publishes random fraudulent transaction data in real-time.
- `zeromq_consumer`: Receives the published data, evaluates it with the model, and sends alerts via Telegram.

### 5. Telegram Alert System

- `send_telegram_alert`: Sends transaction details to Telegram when a suspicious transaction is detected.

### 6. Additional Analysis Functions

- `compare_fraud_nonfraud`: Compares fraudulent and genuine transactions.
- `plot_fraud_distribution`: Displays the distribution of fraudulent and genuine transactions.
- `fraud_categories`: Lists the transaction types with the most fraud.
- `fraud_trend`: Analyzes monthly fraud trends.
- `monthly_fraud_count`: Lists monthly fraud counts.
- `fraud_rate_by_type`: Calculates the fraud rate by transaction type.
- `correlation_heatmap`: Creates a heatmap showing correlations between features.
- `high_value_alerts`: Identifies high-value transactions and logs them.
