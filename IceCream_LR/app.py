from flask import Flask, render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

app = Flask(__name__)

@app.route('/')
def home():
    df = pd.read_csv('C:/Users/ADMIN/Desktop/Flask_EX/IceCream_LR/Ice Cream Sales - temperatures.csv')
    X = df[["Temperature"]]
    y = df[["Ice Cream Profits"]]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = LinearRegression()

    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)

    men= r2_score(y_test, y_pred)*100
    mae= mean_absolute_error(y_test, y_pred)
    mea2= mean_squared_error(y_test, y_pred)

    return render_template('home.html', var=y_pred, mae=mae, r2=men, mse=mea2)

if __name__ == '__main__':
    app.run(debug=True)