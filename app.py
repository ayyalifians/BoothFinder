from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# --- 1. LOAD MODEL ---
print("Memuat model machine learning...")
try:
    with open('model_rekomendasi.pkl', 'rb') as f:
        data = pickle.load(f)
        df_tenant = data['df_tenant']
        tfidf_matrix = data['tfidf_matrix']
        tfidf = data['tfidf_model']
    print("Model berhasil dimuat!")
except Exception as e:
    print(f"Gagal memuat model: {e}")

# --- 2. ROUTING FRONTEND (SUDAH DIMODIFIKASI UNTUK KPI CARD) ---
@app.route('/')
def home():
    try:
        # 1. Hitung jumlah tenant unik secara dinamis dari model yang dimuat
        jumlah_tenant = len(df_tenant)
        
        # 2. Definisikan jumlah produk mentah (Hardcoded untuk menghemat RAM server)
        # Angka ini kita kirim ke index.html agar dinamis di sisi frontend
        jumlah_produk = 15232
        
        # Silakan ubah angka pembagian di bawah ini sesuai dengan proporsi data mentah aslimu!
        # (Misal: dari 15.232, berapa F&B dan berapa Fashion)
        produk_fb = 14500      # Ganti dengan angka aslinya
        produk_fashion = 732   # Ganti dengan angka aslinya

        # Lempar variabel ke Jinja2 HTML
        return render_template('index.html', 
                               jumlah_tenant=jumlah_tenant,
                               jumlah_produk=jumlah_produk,
                               produk_fb=produk_fb,
                               produk_fashion=produk_fashion)
                               
    except Exception as e:
        print("ERROR ROUTING HOME:", str(e))
        return render_template('index.html') # Fallback jika terjadi error

# --- 3. ROUTING API PENCARIAN ---
@app.route('/api/recommend', methods=['POST'])
def recommend():
    try:
        req_data = request.get_json()
        user_query = req_data.get('query', '')
        price_limit = req_data.get('price_limit', '')
        sektor_filter = req_data.get('sektor', 'Semua')

        if not user_query:
            return jsonify({'error': 'Kriteria pencarian tidak boleh kosong!'})

        query_vec = tfidf.transform([user_query.lower()])
        cosine_sim = cosine_similarity(query_vec, tfidf_matrix).flatten()
        
        results = df_tenant.copy()
        results['score'] = cosine_sim
        
        if price_limit:
            results = results[pd.to_numeric(results['harga'], errors='coerce') <= float(price_limit)]
            
        if sektor_filter != 'Semua':
            results = results[results['sektor'] == sektor_filter]
            
        final_results = results[results['score'] > 0].sort_values(by='score', ascending=False).head(10)
        output = final_results[['nama_tenant', 'sektor', 'harga', 'score']].to_dict(orient='records')
        
        return jsonify({'recommendations': output})
    except Exception as e:
        print("ERROR RECOMMEND:", str(e))
        return jsonify({'error': str(e)})

# --- 4. ROUTING API STATISTIK EDA ---
@app.route('/api/stats', methods=['GET'])
def get_stats():
    try:
        sektor_counts = df_tenant['sektor'].value_counts().to_dict()
        
        df_tenant['harga_num'] = pd.to_numeric(df_tenant['harga'], errors='coerce')
        bins = [0, 25000, 50000, 100000, float('inf')]
        labels = ['< Rp25k', 'Rp25k - 50k', 'Rp50k - 100k', '> Rp100k']
        df_tenant['rentang_harga'] = pd.cut(df_tenant['harga_num'], bins=bins, labels=labels)
        price_dist = df_tenant['rentang_harga'].value_counts().reindex(labels).fillna(0).to_dict()
        
        # Ambil 40 data kombinasi
        preview_fb = df_tenant[df_tenant['sektor'].str.contains('F&B', case=False, na=False)].head(20)
        preview_fashion = df_tenant[df_tenant['sektor'].str.contains('Fashion', case=False, na=False)].head(20)
        
        # Pencegahan error jika nama sektor di dataset berbeda
        if preview_fb.empty and preview_fashion.empty:
            preview_combined = df_tenant.head(40)
        else:
            preview_combined = pd.concat([preview_fb, preview_fashion])
            
        preview_data = preview_combined[['nama_tenant', 'sektor', 'harga']].to_dict(orient='records')
        
        return jsonify({
            'sektor': sektor_counts,
            'harga_dist': price_dist,
            'preview': preview_data
        })
    except Exception as e:
        print("ERROR STATS:", str(e))
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)