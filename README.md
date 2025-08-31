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
- Jheison Gomez Muñoz - 2310215

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

twitter_sentiment_analysis/
│
├── data/
│   ├── raw/               # Original dataset (unchanged) https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment/data
│   └── processed/         # Preprocessed data (tokenized, padded, encoded)
│
├── notebooks/
│   └── exploration.ipynb  # Exploratory analysis and initial tests
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py    # Cleaning, tokenization, padding
│   ├── models.py                # Model architectures (MLP, RNN, LSTM/GRU)
│   ├── train.py                 # Training and evaluation functions
│   └── utils.py                 # Helper functions (plot metrics, save/load models)
│
├── outputs/
│   ├── models/                  # Model checkpoints and weights
│   └── figures/                 # Plots, learning curves, results
│
├── README.md                     # Project documentation (optional)
├── requirements.txt              # Python dependencies
├── .gitignore                    # Files to ignore in Git
└── main.py                       # Main script to run preprocessing, training, evaluation

===========================================================
INSTALLATION & SETUP
===========================================================

1. Clone the repository:
   git clone <repository_url>
2. Navigate to the project folder:
   cd twitter_sentiment_analysis
3. Create a virtual environment (recommended):
   python -m venv env
4. Activate the environment:
   - Windows: env\Scripts\activate
   - Linux/macOS: source env/bin/activate
5. Install dependencies:
   pip install -r requirements.txt

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

===========================================================
USAGE
===========================================================

Run the main script to execute the preprocessing, training, and evaluation pipeline:

   python main.py

Optional: Run exploratory notebooks to inspect and visualize the dataset:

   jupyter notebook notebooks/exploration.ipynb

The outputs (trained models, evaluation plots) will be saved in the 'outputs/' folder.

===========================================================
AUTHOR
===========================================================

- Valentina Barbetty Arango - 2310050
- Brayan Gomez Muñoz - 2310016
- Jheison Gomez Muñoz - 2310215

===========================================================
LICENSE
===========================================================

[Specify license if needed, e.g., MIT]
