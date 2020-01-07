import math
import pandas as pd
import numpy as np
from flask import Flask, jsonify, request, render_template
import pickle

# load model
model = pickle.load(open('model.pkl','rb'))

# app
app = Flask(__name__)

# routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])

def predict():
    # # get data
    # data = request.get_json(force=True)

    # # convert data into dataframe
    # data.update((x, [y]) for x, y in data.items())
    # data_df = pd.DataFrame.from_dict(data)

    # # predictions
    # result = model.predict(data_df)

    # # send back to browser
    # output = {'results': int(result[0])}

    # # return data
    # return jsonify(results=output)

    data = [int(x) for x in request.form.values()]
    final_features = [np.array(data)]
    prediction = model.predict(final_features)
    output = round(prediction[0], 2)
    # output = {'results': int(prediction[0])}
    if(output == 1):
        result = 'berhasil'
    else:
        result = 'gagal'
    
    return render_template('index.html', prediction_text='Kamu {}'.format(result))
    # return jsonify(results=output)

if __name__ == '__main__':
    app.run(port = 80, debug=false)

