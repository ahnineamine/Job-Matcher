from gensim.models.keyedvectors import KeyedVectors
import json
import os
import io
import unicodedata
import flask
from flask import request, jsonify
from DocSim import DocSim

#Initiate Flask application
app = flask.Flask(__name__)

#load word embeddings model
model_path = "./models/word2vec_model_optim"
model = KeyedVectors.load(model_path)

#initiate DocSim object
ds = DocSim(model)

@app.route('/processjson',methods=['POST'])
def process_files():
    data = request.get_json()
    L_input=[]
    L_index=[]
    for cv in data['cv']:
        index = cv['id']
        content = cv['info']
        L_input.append(content)   
        L_index.append(index)

    for job in data['job']:
        job_content = job['jobinfo']
        L_input.append(job_content)
        
    #return jsonify(L_input)
    return jsonify(ds.calculate_similarity(L_input,L_index))

if __name__ == '__main__':
   port = int(os.environ.get("PORT", 5000))
   app.run(host='0.0.0.0', port=port, debug=True)

