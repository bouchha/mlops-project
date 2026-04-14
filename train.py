from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

# Charger dataset
data = load_iris()
X = data.data
y = data.target

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Modèle
model = LogisticRegression(max_iter=200)

# Entraînement
model.fit(X_train, y_train)

# Évaluation
accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)

# Sauvegarde
joblib.dump(model, "model.pkl")