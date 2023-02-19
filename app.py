from flask import Flask, render_template, request, jsonify
import joblib
import pickle
import numpy as np

# load the machine learning model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# define a function to do the prediction
def predict(data):
    # apply pre-processing steps to the data
    data = np.array(data).reshape(1, -1)

    # make the prediction using the machine learning model
    prediction = model.predict(data)

    return prediction[0]
app = Flask(__name__, template_folder='templates')


# Define the route for the user interface
@app.route('/')
def index():
    return render_template('templates/index.html')

# Define the API endpoint for making a prediction
@app.route('/make-prediction', methods=['POST'])
def make_prediction():
    # get the input data from the request
    data = request.get_json()

    # make the prediction
    prediction = predict(data)

    # return the prediction as a JSON response
    response = jsonify({'prediction': prediction.tolist()})
    return response

if __name__ == '__main__':
    app.run()
