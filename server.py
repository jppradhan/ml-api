from flask import Flask, request, jsonify
import traceback
import pandas as pd
# import numpy as np
import pickle

app = Flask(__name__)


@app.route('/predict', methods=['GET'])
def predict():
    if lr:
        try:
            json_ = request.json
            print(json_)
            #query = pd.get_dummies(pd.DataFrame(json_))
            #query = query.reindex(columns=model_columns, fill_value=0)

            future = lr.make_future_dataframe(periods=365)
            forecast = lr.predict(future)
            prediction = forecast[['ds', 'yhat']]

            df = pd.DataFrame(prediction)

            return jsonify({'prediction': df.to_json(orient='split')})

        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print('Train the model first')
        return ('No model here to use')


if __name__ == '__main__':
    try:
        port = int(sys.argv[1])  # This is for a command-line input
    except:
        port = 5000  # If you don't provide any port the port will be set to 12345

    with open('./src/bikes.pkl', 'rb') as fout:
        lr = pickle.load(fout)  # Load "model.pkl"
    print('Model loaded')

    app.run(port=port, debug=True)
