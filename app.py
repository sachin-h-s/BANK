# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 17:11:14 2021

@author: sachin h s
"""

from flask import Flask, render_template, request
import pickle
import numpy as np


model = pickle.load(open('classifier.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def man():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def home():
    data1 = request.form['variance']
    data2 = request.form['skewness']
    data3 = request.form['curtosis']
    data4 = request.form['entropy']
    arr = np.array([[data1, data2, data3, data4]])
    pred = model.predict(arr)
    return render_template('after.html', data=pred)


if __name__ == "__main__":
    app.run()