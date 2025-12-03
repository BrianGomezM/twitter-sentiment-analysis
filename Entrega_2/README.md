===========================================================
        TWITTER US AIRLINE SENTIMENT ANALYSIS
===========================================================

Universidad del Valle
Facultad de Ingeniería
Asignatura: Redes Neuronales - 2025
Profesora: Deisy Chaves

Estudiantes:
- Valentina Barbetty Arango - 2310050
- Brayan Gomez Muñoz - 2310016

Proyecto: Twitter US Airline Sentiment Dataset

===========================================================
INTRODUCTION
===========================================================

This project focuses on sentiment analysis of US airline tweets using neural networks.
The main objective is to classify tweets as positive, negative, or neutral.
We explore different architectures including:

- MLP (Multi-Layer Perceptron)
- RNN (Recurrent Neural Network)
- LSTM/GRU (RNNs with memory)

The project is implemented in Python using TensorFlow/Keras and other supporting libraries.
It is modularized for easy extension and maintenance.

===========================================================
PROJECT STRUCTURE
===========================================================
├── src/
│   ├── preprocess.py
│   ├── train_rnn.py
│   ├── train_lstm.py
│   ├── utils.py
│   └── evaluation.py
├── main.py
├── requirements.txt
└── README.md

===========================================================
INSTALLATION & SETUP
===========================================================

1. Clone the repository:
    git clone https://github.com/BrianGomezM/twitter-sentiment-analysis
2. Navigate to the project folder:
    cd twitter_sentiment_analysis
3. Create a virtual environment (recommended):
     py -3.10 -m venv venv   
4. Activate the environment:
    Windows: .\venv\Scripts\Activate.ps1 
    Linux/macOS: source env/bin/activate
5. Install dependencies:
    pip install -r requirements.txt
6. Ejecución
    py main.py  : El lstm
    py main_rnn.py   : Rnn normal  
7. Ejecutar modelos
    py .\src\predict_sentiment.py

===========================================================
REQUIREMENTS / LIBRARY VERSIONS
===========================================================

- Python 3.10+
- numpy >= 1.23
- pandas >= 2.0
- scikit-learn >= 1.2
- matplotlib >= 3.7
- seaborn >= 0.12
- tensorflow >= 2.13
- keras >= 2.13
- jupyter >= 1.0