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
#app.config["DEBUG"] = True

#load word embeddings model
model_path = "./models/word2vec_model_optim"
model = KeyedVectors.load(model_path)

#initiate DocSim object
ds = DocSim(model)
"""
---this part is for Json file input---

L_input=[]
L_index=[]
data = json.loads(open('test-data.json').read()) #name of json file in general
for cv in data['cv']:
    index = cv['id']
    content = cv['exp']+','+ cv['skill']
    L_input.append(content)   
    L_index.append(index)

content_offer = data['job']['title']+','+data['job']['desc']+','+data['job']['req']
L_input.append(content_offer)

"""
def get_data(file):
    with open(file) as f:
        line_list=[]
        index_list = []
        for index,l in enumerate(f):
            line = l.rstrip('\n')
            line_uni = unicodedata.normalize("NFKD",line)
            line_list.append(line_uni)
            index_list.append(index)
    return line_list,index_list


#L_input,L_index= get_data('data/validation_data/validation_set_ITT.csv')
@app.route('/')
def index():
    return 'Hello World!'

if __name__ == '__main__':
   port = int(os.environ.get("PORT", 5000))
   app.run(host='0.0.0.0', port=port, debug=True)

#sim_scores = ds.calculate_similarity(L_input,L_index)

#print(sim_scores)

