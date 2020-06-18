import json

from flask import Flask, jsonify
from flask_cors import CORS
from flask import request

from MoraCategoryClassifier import MoraCategoryClassifier

application = Flask(__name__)
CORS(application)
ml_models = MoraCategoryClassifier()



@application.route('/health', methods=['GET'])
def pong():
    return jsonify({
        'health': 'awesome'
    })


@application.route('/signals_mltool/predict', methods=['POST'])
def predict():
    query = json.loads(request.data)['text']

    pred_hoofdrubriek = ml_models.classifyAllCategoriesWithProbability(query)
    pred_subrubriek = ml_models.classifyAllSubCategoriesWithProbability(query)

    return jsonify({'hoofdrubriek': list(pred_hoofdrubriek),
                    'subrubriek': list(pred_subrubriek)})


if __name__ == '__main__':
    application.run(port=8000)
