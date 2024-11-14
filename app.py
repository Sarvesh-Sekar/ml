from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)
CORS(app)

# Sample data
data = {
    'customer_id': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
    'product_id': [101, 102, 101, 103, 104, 105, 102, 102, 102, 106],
    'purchase_date': ['2023-01-01', '2023-01-05', '2023-02-01', '2023-02-10', '2023-03-01', '2023-03-05', '2023-04-01', '2023-04-10', '2023-04-10', '2023-04-10']
}

df = pd.DataFrame(data)
df['purchase_date'] = pd.to_datetime(df['purchase_date'])

product_mapping = {
    101: 'Product A',
    102: 'Product B',
    103: 'Product C',
    104: 'Product D',
    105: 'Product E',
    106: 'Product F'
}

customer_encoder = LabelEncoder()
product_encoder = LabelEncoder()

df['customer_encoded'] = customer_encoder.fit_transform(df['customer_id'])
df['product_encoded'] = product_encoder.fit_transform(df['product_id'])

user_item_matrix = csr_matrix((df['purchase_date'].notnull().astype(int),
                               (df['customer_encoded'], df['product_encoded'])))

knn = NearestNeighbors(metric='cosine', algorithm='brute')
knn.fit(user_item_matrix)

@app.route('/recommend', methods=['GET'])
def recommend():
    customer_id = int(request.args.get('customer_id'))
    n_neighbors = min(5, user_item_matrix.shape[0])
    customer_encoded = customer_encoder.transform([customer_id])[0]
    distances, indices = knn.kneighbors(user_item_matrix[customer_encoded], n_neighbors=n_neighbors)

    # Get products bought by this customer
    customer_products = df[df['customer_encoded'] == customer_encoded]['product_id'].values

    # Find the most frequently bought product by this customer
    if len(customer_products) > 0:
        most_bought_product = pd.Series(customer_products).mode()[0]
    else:
        most_bought_product = None

    # Get products bought by similar customers
    similar_customers = indices.flatten()
    similar_products = df[df['customer_encoded'].isin(similar_customers)]['product_id'].values

    # Recommend products based on similar customers excluding already bought products by the user
    recommended_products = [product for product in similar_products if product != most_bought_product]
    recommended_products = list(set(recommended_products))

    if most_bought_product:
        recommended_products.insert(0, most_bought_product)

    recommended_product_names = [product_mapping[pid] for pid in recommended_products]

    return jsonify(recommended_product_names)

if __name__ == '__main__':
    app.run(debug=True)
