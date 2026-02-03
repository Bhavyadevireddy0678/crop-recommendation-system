# crop-recommendation-system
# Crop Recommendation System

A machine learning system that recommends the best crop to grow based on soil composition and weather conditions. Trained on an agricultural dataset and benchmarked across multiple classifiers.

## How It Works

1. Takes soil and weather parameters as input (e.g. nitrogen, phosphorus, potassium, temperature, humidity, rainfall, pH)
2. Runs the input through a trained Random Forest classifier
3. Returns a ranked list of crop recommendations based on the input conditions

## Tech Stack

- **Python** — core language
- **Scikit-learn** — model training, evaluation, and prediction
- **Pandas** — data loading and preprocessing
- **NumPy** — numerical computations and feature engineering

## Results

- Benchmarked Random Forest, KNN, and SVM classifiers
- Optimized Random Forest achieved ~90% accuracy
- Real-time prediction module for user-provided inputs

## How to Run

1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/crop-recommendation-system.git
cd crop-recommendation-system
```

2. Install dependencies
```bash
pip install pandas numpy scikit-learn
```

3. Run the main script
```bash
python main.py
```

## Project Structure

```
crop-recommendation-system/
├── main.py              # Entry point and prediction module
├── requirements.txt     # Python dependencies
└── README.md            # This file
```
