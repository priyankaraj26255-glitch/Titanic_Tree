"""Load the exported model and predict for one passenger."""
import joblib
import pandas as pd
from pathlib import Path

bundle = joblib.load(Path(__file__).parent / "models" / "titanic_tree.joblib")
model, columns = bundle["model"], bundle["columns"]

# Example: 28-year-old woman, 1st class, embarked at Cherbourg
passenger = {"pclass": 1, "sex": 1, "age": 28, "sibsp": 0, "parch": 0,
             "fare": 80.0, "emb_C": 1, "emb_Q": 0, "emb_S": 0}
X = pd.DataFrame([passenger]).reindex(columns=columns, fill_value=0)
print("Survived" if model.predict(X)[0] == 1 else "Died",
      model.predict_proba(X)[0].round(3))
