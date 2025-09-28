from sklearn.metrics import classification_report

def evaluate_model(model, X_test, y_test, encoder):
    y_pred = model.predict(X_test.toarray()) #Predecir en conjunto de test
    y_pred_classes = y_pred.argmax(axis=1) #Selecciona la clase con mayor probabilidad
    report = classification_report(y_test, y_pred_classes, target_names=encoder.classes_) #Genera reporte de clasificación
    print(report)
