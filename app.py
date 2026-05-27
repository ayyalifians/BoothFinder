from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# 1. Buka dan muat "Otak" Machine Learning (File Pickle)
print("Memuat model machine learning...")
with open('model_rekomendasi.pkl', 'rb') as f:
    data = pickle.load(f)
    df_tenant = data['df_tenant']
    tfidf_matrix = data['tfidf_matrix']
    tfidf = data['tfidf_model']
print("Model berhasil dimuat!")

# 2. Route Halaman Utama (Wajah Web)
@app.route('/')
def home():
    return render_template('index.html')

# 3. Route API untuk memproses pencarian panitia
@app.route('/api/recommend', methods=['POST'])
def recommend():
    # Menangkap data yang dikirim dari form HTML
    req_data = request.get_json()
    user_query = req_data.get('query', '')
    price_limit = req_data.get('price_limit', '')
    sektor_filter = req_data.get('sektor', 'Semua')

    if not user_query:
        return jsonify({'error': 'Kriteria pencarian tidak boleh kosong!'})

    try:
        # Proses 1: Ubah teks pencarian ke vektor
        query_vec = tfidf.transform([user_query.lower()])
        
        # Proses 2: Hitung kemiripan (Cosine Similarity)
        cosine_sim = cosine_similarity(query_vec, tfidf_matrix).flatten()
        
        # Proses 3: Masukkan skor ke data
        results = df_tenant.copy()
        results['score'] = cosine_sim
        
        # Proses 4: Filter Budget (Jika diisi)
        if price_limit:
            results = results[results['harga'] <= float(price_limit)]
            
        # Proses 5: Filter Sektor
        if sektor_filter != 'Semua':
            results = results[results['sektor'] == sektor_filter]
            
        # Ambil Top 10 skor tertinggi (di atas 0)
        final_results = results[results['score'] > 0].sort_values(by='score', ascending=False).head(10)
        
        # Format hasil menjadi list of dictionary agar bisa dibaca JavaScript
        output = final_results[['nama_tenant', 'sektor', 'harga', 'score']].to_dict(orient='records')
        
        return jsonify({'recommendations': output})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # Jalankan server di mode debug
    app.run(debug=True)