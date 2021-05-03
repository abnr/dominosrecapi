from flask import Flask, request
from flask_restful import reqparse, abort, Api, Resource
import pickle
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


app = Flask(__name__)
api = Api(app)


clf_path = 'knn.pkl'
model = []
with open(clf_path, 'rb') as f:
    model = pickle.load(f)

apriori = []
with open('apriori_sobremesas.pkl', 'rb') as f:
    apriori = pickle.load(f)

class upgradepizza(Resource):

	def post(self):
		df = model[1]
		df2 = df.iloc[9:, :]
		data = request.get_json(force=True)
		if len(data['pizzas']) == 0:
			return {'suggestao':0}
		
		pizzas = []
		for p in data['pizzas']:
			if p in list(df.index[0:9]):
				pizzas.append(p)
				
		recomendacao = []
		for p in pizzas:
			resp = model[0].kneighbors( df.loc[p].values.reshape(1, -1))
			recomendacao.append( df2.index[resp[1][0][0]] )
			
		return {'sugestao':recomendacao}

def check_lists(list1, list2):

	count = 0
	for elem in list1:
		if elem in list2:
			count += 1
	return count == len(list1)

class sobremesas(Resource):

	def post(self):

		data = request.get_json(force=True)
		if len(data['pizzas']) == 0:
			return {'suggestao':0}
		
		sobremesas_ret = []
		for pair in apriori:
			if len(pair[0]) == len(data['pizzas']):
				result = check_lists( pair[0], data['pizzas'] )
				if result:
					print('pizza ',data['pizzas'])
					print('par ',pair[0])
					sobremesas_ret.append(np.random.choice(pair[1]))
		
			
		return {'sugestao':list(set(sobremesas_ret))}

api.add_resource(upgradepizza, '/uppizza')
api.add_resource(sobremesas, '/sobremesas')


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)

