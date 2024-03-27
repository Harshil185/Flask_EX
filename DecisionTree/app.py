from flask import Flask, render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

app = Flask(__name__)

@app.route('/')
def home():
    df = pd.read_csv('Decision-Tree-Classification-Data.csv')

    X = df.drop('diabetes', axis=1)
    y = df.diabetes

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.28, random_state=43)

    Decision_Tree_Class_Model = DecisionTreeClassifier()
    Decision_Tree_Class_Model.fit(X_train, y_train)
    Y_pred = Decision_Tree_Class_Model.predict(X_test)

    da = accuracy_score(y_test, Y_pred)

# modifing it using different parmeters
    Decision_Tree_Class_Model1 = DecisionTreeClassifier(criterion="entropy",max_depth=7)
    Decision_Tree_Class_Model1.fit(X_train, y_train)
    Y_pred1 = Decision_Tree_Class_Model1.predict(X_test)
    da1 = accuracy_score(y_test, Y_pred1)

    return render_template('home.html', da=da, da1=da1)

if __name__ == '__main__':
    app.run(debug=True)
