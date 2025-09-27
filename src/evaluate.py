from sklearn.metrics import classification_report

def evaluate_model(model, X_test, y_test, encoder):
    y_pred = model.predict(X_test.toarray())
    y_pred_classes = y_pred.argmax(axis=1)
    report = classification_report(y_test, y_pred_classes, target_names=encoder.classes_)
    print(report)
