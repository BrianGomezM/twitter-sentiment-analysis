from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input

def build_mlp(input_dim, num_classes=3):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(512, activation='relu'), #Capa oculta con 512 neuronas y activación ReLU
        Dropout(0.5), #Apagar el 50% de las neuronas para evitar overfitting
        Dense(256, activation='relu'), #Capa oculta con 256 neuronas y activación ReLU
        Dropout(0.5),
        Dense(num_classes, activation='softmax') #Capa de salida con activación softmax para clasificación multiclase
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    ) 
    return model
