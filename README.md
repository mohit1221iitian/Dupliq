# 🔁 Dupliq — Your AI-Powered Duplicate Question Detector

> 💬 “Do these questions mean the same thing?” Let **Dupliq** do the thinking.

**Dupliq** is a smart, efficient, and user-friendly AI tool that identifies whether two questions are semantically similar — even when worded differently. Ideal for Q\&A platforms, search engines, chatbots, or any system that needs to eliminate redundancy.

---

## ✨ Key Features

* 🧠 **Semantic Intelligence** – Uses NLP and embeddings to capture real meaning.
* 🚀 **High Accuracy (81.78%)** – Backed by Word2Vec and Random Forest.
* ⚡ **Real-Time Response** – Just input two questions and get instant feedback.
* 🎯 **Simple Interface** – Powered by Streamlit for a seamless user experience.

---

## 🧠 How It Works

Dupliq uses a combination of:

* Word2Vec embeddings to understand semantic structure
* FuzzyWuzzy for textual similarity
* Random Forest classifier for prediction

🎯 **Test Accuracy**: `81.77%`

---

## ⚙️ Tech Stack

* **Python 3.10**
* **Streamlit** – Interactive frontend
* **scikit-learn** – Model training
* **Gensim** – Word2Vec embeddings
* **FuzzyWuzzy** – Text similarity
* **Pandas & NumPy** – Data processing

---

## 🚀 Getting Started

### 🔧 Clone & Run Locally

```bash
git clone https://github.com/mohit1221iitian/Dupliq.git
cd Dupliq
```

No extra setup required. You're ready to go!

### ▶️ Launch the App

```bash
streamlit run app.py
```

Visit `http://localhost:8501` in your browser.

---

## 📸 Live Preview

> User-friendly interface to check for duplicate questions:

![Dupliq Screenshot](https://github.com/mohit1221iitian/Dupliq/blob/master/screenshot.png)

---

## 📁 Project Structure

```
Dupliq/
├── app.py                # Streamlit app
├── helper.py             # NLP processing and features
├── rf_w2v_model1.pkl     # Trained Random Forest model
├── requirements.txt      # Project dependencies
|- xgb_w2v_model1.pkl     
```

---

## 📄 License

This project is licensed under the **MIT License**. Feel free to use, modify, and share.

---

## 👨‍💻 Author

**Mohit Gunani**
📧 [gunanimohit1221@gmail.com](mailto:gunanimohit1221@gmail.com)
🔗 [GitHub: @mohit1221iitian](https://github.com/mohit1221iitian)

---

## ⭐ Support the Project

If you find Dupliq helpful, please ⭐ the repo and share it with your network. Your support fuels open-source innovation!

---

> Built with ❤️ to make content smarter, cleaner, and more meaningful.
