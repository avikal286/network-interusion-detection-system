# 🛡️ Network Intrusion Detection System (NIDS)

A **Streamlit-based Network Intrusion Detection System** that uses **Deep Learning and Machine Learning** techniques to detect malicious network traffic.  
The system supports **real-time single input analysis**, **batch CSV analysis**, **data visualization**, and **on-the-fly classification training**.

---

## 🚀 Features

### 🔹 Single Traffic Analysis
- Predicts whether a network connection is **NORMAL** or an **ATTACK**
- Uses a **pre-trained Deep Learning model (Keras `.h5`)**
- Probability-based decision with a configurable threshold

### 🔹 Batch CSV Analysis
- Upload large CSV files for analysis
- View:
  - Data preview
  - Descriptive statistics
  - Filtered subsets
- Interactive plots:
  - Line charts
  - Pie charts

### 🔹 ML Classification Module
- Train a **Decision Tree Classifier**
- Automatic:
  - Encoding of categorical features
  - Feature scaling
- Outputs:
  - Accuracy score
  - Classification report
  - Confusion matrix

### 🔹 Modern Dark UI
- Custom CSS
- Clean, dashboard-style interface
- Optimized for wide screens

---

## 🧠 Tech Stack

- **Frontend / UI**: Streamlit
- **Deep Learning**: TensorFlow / Keras
- **Machine Learning**: Scikit-learn
- **Data Handling**: Pandas, NumPy
- **Visualization**: Matplotlib
- **Model Artifacts**: Pickle

---

## 📂 Project Structure

├── app.py # Main Streamlit application
├── nids_model.h5 # Trained deep learning model
├── scaler.pkl # StandardScaler object
├── encoders.pkl # LabelEncoders for categorical features
├── features.pkl # Ordered feature list
├── requirements.txt # Python dependencies
└── README.md # Project documentation

yaml

## 📊 Dataset

- Trained on the **KDD Cup Network Intrusion Dataset**
- Supports common network attributes such as:
  - protocol_type
  - service
  - flag
  - src_bytes
  - dst_bytes
  - connection counts

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
3️⃣ Install Dependencies
bash
pip install -r requirements.txt
4️⃣ Update Model Paths
Make sure paths to these files are correct in the code:

nids_model.h5

scaler.pkl

encoders.pkl

features.pkl

▶️ Run the Application
bash
streamlit run app.py
The app will open automatically in your browser.

🧪 How Prediction Works
User inputs or uploads network traffic data

Categorical features are encoded

Numeric features are scaled

Deep Learning model predicts attack probability

Output:

NORMAL ✅ or ATTACK 🚨

📈 Threshold Logic
python
threshold = 0.2
label = "ATTACK" if probability > threshold else "NORMAL"
You can adjust the threshold based on security requirements.

🛠️ Future Improvements
Support for real-time packet capture

Additional ML models (Random Forest, XGBoost)

Model performance comparison dashboard

Deployment on cloud platforms

👤 Author
Avikal Bhatt
GitHub: https://github.com/avikal286

LinkedIn: https://www.linkedin.com/in/avikal-bhatt-418902372

📜 License
This project is for educational and research purposes.

⭐ If you find this project useful, consider giving it a star on GitHub!
