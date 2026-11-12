# 🐑 Sheep Breed Analysis Using Deep Learning

An AI-powered Sheep Breed Classification application that uses Deep Learning and Convolutional Neural Networks (CNN) to identify sheep breeds from uploaded images.

The application allows users to upload a sheep image and receive a predicted breed, prediction confidence, breed information, typical weight range, and prediction probabilities.

## 🚀 Live Demo

**Streamlit Application:**  
https://sheep-breed-analysis-a3jaecq5a2yappr9ovilzpt.streamlit.app/

## 📌 Project Overview

Sheep Breed Analysis is a machine learning application developed to automate sheep breed identification using image classification techniques.

The project uses a trained deep learning model to analyze uploaded sheep images and predict their corresponding breeds.

It provides useful information about the predicted breed and displays prediction confidence to help users understand the model's results.

## ✨ Features

- Upload sheep images for classification.
- AI-powered sheep breed prediction.
- Prediction confidence score.
- Breed information and description.
- Typical sheep weight range.
- Prediction probability visualization.
- Simple and user-friendly Streamlit interface.
- Real-time image classification.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| TensorFlow | Deep Learning framework |
| Keras | Model development and loading |
| CNN | Image classification |
| NumPy | Numerical operations |
| Pandas | Data handling |
| Matplotlib | Data visualization |
| Pillow | Image processing |
| Streamlit | Web application |
| Git & GitHub | Version control |

## 🧠 Deep Learning Model

The project uses a trained CNN-based image classification model.

**Model file:** `new_sheep_model.keras`

The model processes the uploaded image and predicts the most likely sheep breed based on the learned image features.

## 📂 Project Structure

```text
sheep-breed-analysis/
│
├── app.py
├── new_sheep_model.keras
├── requirements.txt
├── runtime.txt
├── README.md
├── .gitattributes
│
├── SheepFaceImages/
│
├── static/
│
├── test2.jpg
└── test3.jpg
```

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/venugopalats58-ctrl/sheep-breed-analysis.git
```

### 2. Navigate to the Project Folder

```bash
cd sheep-breed-analysis
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Environment

Windows:

```bash
venv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser at:

`http://localhost:8501`

## 🖼️ How to Use

1. Open the Sheep Breed Analysis application.
2. Upload an image of a sheep.
3. Wait for the AI model to process the image.
4. View the predicted sheep breed.
5. Check prediction confidence.
6. Explore breed information and prediction probabilities.

## ☁️ Deployment

The application is deployed using **Streamlit Community Cloud**.

Live application:

https://sheep-breed-analysis-a3jaecq5a2yappr9ovilzpt.streamlit.app/

## 🎯 Project Objective

The main objective is to apply Artificial Intelligence and Deep Learning techniques to automate sheep breed identification through image classification.

This project demonstrates the practical use of computer vision and deep learning in livestock management.

## 🔮 Future Enhancements

- Support for more sheep breeds.
- Improve prediction accuracy.
- Add sheep health monitoring features.
- Integrate sheep weight estimation.
- Develop a mobile-friendly application.
- Add breed comparison and detailed analytics.

## 👨‍💻 Developer

**Venugopala T S**

BCA Student | Aspiring AI & ML Engineer

- GitHub: https://github.com/venugopalats58-ctrl
- LinkedIn: https://linkedin.com/in/venugopala-t-s-943bb1349/

---

⭐ If you find this project interesting, consider giving the repository a star!

**Built with Python, Deep Learning, TensorFlow and Streamlit.**
