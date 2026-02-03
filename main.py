import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


# ─── Generate Sample Dataset ───
def generate_crop_data(n_samples=3000, random_state=42):
    """Generate a synthetic agricultural dataset with soil and weather attributes."""
    np.random.seed(random_state)

    data = {
        "nitrogen": np.random.randint(0, 150, n_samples),
        "phosphorus": np.random.randint(0, 150, n_samples),
        "potassium": np.random.randint(0, 150, n_samples),
        "temperature": np.round(np.random.uniform(6, 45, n_samples), 1),
        "humidity": np.round(np.random.uniform(10, 100, n_samples), 1),
        "ph": np.round(np.random.uniform(3.5, 9.0, n_samples), 2),
        "rainfall": np.round(np.random.uniform(20, 300, n_samples), 1),
    }

    # Assign crops based on simple rules (simulates real patterns)
    crops = []
    for i in range(n_samples):
        n, p, k = data["nitrogen"][i], data["phosphorus"][i], data["potassium"][i]
        temp, humid, ph = data["temperature"][i], data["humidity"][i], data["ph"][i]

        if n > 80 and temp > 25 and humid > 60:
            crops.append("Rice")
        elif k > 80 and temp > 20 and ph < 7:
            crops.append("Wheat")
        elif p > 80 and humid < 50 and temp > 30:
            crops.append("Mango")
        elif n > 60 and ph > 6 and humid > 50:
            crops.append("Maize")
        elif temp < 20 and humid > 40 and ph < 6.5:
            crops.append("Apple")
        elif k > 60 and temp > 25 and humid > 70:
            crops.append("Banana")
        elif n < 40 and temp > 28 and humid < 40:
            crops.append("Coconut")
        else:
            crops.append("Grapes")

    data["crop"] = crops
    return pd.DataFrame(data)


# ─── Preprocessing ───
def preprocess(df):
    """Split features and label."""
    X = df.drop(columns=["crop"])
    y = df["crop"]
    return X, y


# ─── Train and Evaluate Models ───
def train_and_evaluate(X_train, X_test, y_train, y_test):
    """Train Random Forest, KNN, and SVM. Print accuracy for each."""
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "SVM": SVC(kernel="rbf", random_state=42),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = {"model": model, "accuracy": round(acc * 100, 2)}
        print(f"  {name}: {round(acc * 100, 2)}% accuracy")

    return results


# ─── Prediction Module ───
def predict_crop(model, input_params):
    """
    Predict the best crop based on user input.
    input_params: dict with keys —
        nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall
    """
    input_df = pd.DataFrame([input_params])
    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    classes = model.classes_

    # Rank crops by probability
    ranked = sorted(zip(classes, probabilities), key=lambda x: x[1], reverse=True)
    return prediction, ranked


def print_prediction(prediction, ranked):
    """Print the prediction result in a readable format."""
    print("\n" + "=" * 50)
    print("  CROP RECOMMENDATION")
    print("=" * 50)
    print(f"\n  Best Crop: {prediction}")
    print("\n  All Recommendations (ranked):")
    for crop, prob in ranked:
        bar = "█" * int(prob * 30)
        print(f"    {crop:<12} {bar} {round(prob * 100, 1)}%")
    print("\n" + "=" * 50 + "\n")


# ─── Main ───
if __name__ == "__main__":
    # 1. Generate data
    print("\nGenerating dataset...")
    df = generate_crop_data(n_samples=3000)
    print(f"  Dataset shape: {df.shape}")
    print(f"  Crops: {df['crop'].value_counts().to_dict()}\n")

    # 2. Preprocess
    X, y = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Train and compare models
    print("Training models...\n")
    results = train_and_evaluate(X_train, X_test, y_train, y_test)

    # 4. Pick the best model
    best_name = max(results, key=lambda k: results[k]["accuracy"])
    best_model = results[best_name]["model"]
    print(f"\n  Best model: {best_name} ({results[best_name]['accuracy']}% accuracy)\n")

    # 5. Sample prediction
    sample_input = {
        "nitrogen": 90,
        "phosphorus": 40,
        "potassium": 50,
        "temperature": 28.5,
        "humidity": 70.0,
        "ph": 6.5,
        "rainfall": 150.0,
    }

    print("Running prediction with sample input:")
    for k, v in sample_input.items():
        print(f"    {k}: {v}")

    prediction, ranked = predict_crop(best_model, sample_input)
    print_prediction(prediction, ranked)
