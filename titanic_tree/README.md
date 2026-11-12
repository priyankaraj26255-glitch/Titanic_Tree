# Titanic Survival Prediction - Decision Tree

## Run
```bash
pip install -r requirements.txt
python main.py      # preprocess, train, plot, evaluate, export
python predict.py   # use the exported model
```

## Output
- `outputs/decision_tree.png` - tree visualization (plot_tree)
- `models/titanic_tree.joblib` - exported model
- Test accuracy is printed in the terminal

## Structure
```
titanic_tree/
  data/titanic.csv
  models/
  outputs/
  main.py
  predict.py
  requirements.txt
```
