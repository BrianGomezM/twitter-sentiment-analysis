from models import build_mlp

def train_model(X_train, y_train, X_val, y_val):
    model = build_mlp(X_train.shape[1]) #Pasa al modelo el número de características (input_dim)
    history = model.fit(
        X_train.toarray(), y_train,
        validation_data=(X_val.toarray(), y_val),
        epochs=10,
        batch_size=32
    )
    return model, history
