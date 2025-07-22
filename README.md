# BigMart Sales Prediction

This project predicts how much each product will sell at different BigMart stores.

We use a machine learning model to look at things like:
- What the product is
- How much it costs
- Where it’s sold
- How big the store is
- And more...

##  Files in this project

- `train.csv` – Old sales data (used to train the model)
- `test.csv` – New data (we need to predict sales for this)
- `submission.csv` – Our final predictions
- `main.py` – The code that cleans data, trains the model, and creates predictions

##  What the code does

1. Loads the data
2. Fills missing values
3. Fixes weird values (like 0 visibility)
4. Adds new useful columns (like how old a store is)
5. Turns words into numbers so the model can understand
6. Trains a model using Random Forest
7. Predicts sales for the test data
8. Saves the predictions to `submission.csv`

## Tools used

- Python
- pandas
- numpy
- scikit-learn

## Submission format

Your predictions should look like this:

| Item_Identifier | Outlet_Identifier | Item_Outlet_Sales |
|------------------|-------------------|-------------------|
| FDA15            | OUT049            | 1798.39           |
| DRC01            | OUT018            | 765.67            |

##  How to run

Make sure all CSV files are in the same folder, then run:

```bash
python main.py
