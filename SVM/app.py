from flask import Flask, render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score

app = Flask(__name__)

@app.route('/')
def home():
    df = pd.read_csv('diabetes.csv')
    shape = df.shape

    X = df.drop("Outcome", axis=1)
    y = df[["Outcome"]].values.ravel()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=39)

# Linear Kernel
    #Create a svm Classifier
    clf = svm.SVC(kernel= 'linear') 
    #Train the model using the training sets
    clf.fit(X_train, y_train)
    #Predict the response for test dataset
    y_pred = clf.predict(X_test)

    la = accuracy_score(y_test, y_pred)
    lp = precision_score(y_test, y_pred)
    lr = recall_score(y_test, y_pred)

# Radial basis function (RBF) kernel
    clf1 = svm.SVC(kernel='rbf')
    clf1.fit(X_train, y_train)
    y_pred1 = clf1.predict(X_test)

    rbf_a = accuracy_score(y_test, y_pred1)

# Polynomial kernel
    clf2 = svm.SVC(kernel= "poly")
    clf2.fit(X_train, y_train)
    y_pred2 = clf2.predict(X_test)

    pa = accuracy_score(y_test, y_pred2)
    
    return render_template('home.html', shape=shape, la=la, lp=lp, lr=lr, rbf_a=rbf_a, pa=pa)

if __name__ == '__main__':
    app.run(debug=True)