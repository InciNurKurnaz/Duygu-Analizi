import streamlit as st
import pandas as pd
import sys
import os
from pathlib import Path
from tensorflow.keras.models import load_model
from gensim.models import Word2Vec
import numpy as np
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
import plotly.express as px

# ===================== KLASÖR YAPISI =====================

# Bu dosya: DuyguAnalizi/VeriToplamaVeArayuz/gui.py
CURRENT_DIR = Path(__file__).resolve().parent  # VeriToplamaVeArayuz/
BASE_DIR = CURRENT_DIR.parent  # DuyguAnalizi/

# MetinSiniflandirma klasörü
METIN_SINIF_DIR = BASE_DIR / "MetinSiniflandirma"
MODELS_DIR = METIN_SINIF_DIR / "models"
DATA_DIR = BASE_DIR / "data"

# exported_csv klasörünü oluştur
SAVE_DIR = CURRENT_DIR / "exported_csv"
SAVE_DIR.mkdir(exist_ok=True)

# MetinSiniflandirma'yı path'e ekle
sys.path.insert(0, str(METIN_SINIF_DIR))

print("CURRENT_DIR:", CURRENT_DIR)
print("BASE_DIR:", BASE_DIR)
print("METIN_SINIF_DIR:", METIN_SINIF_DIR)
print("sys.path before:", sys.path)
print("sys.path after:", sys.path)
print("Dosyalar:", os.listdir(METIN_SINIF_DIR))


# ===================== MODÜLLER =====================

from data_fetcher import get_video_id_from_url, get_video_comments

# preprocessing_word2vec.py'den fonksiyonları import et
try:
    from preprocessing_word2vec import preprocess_text, document_vector_hybrid
except ImportError:
    st.error("❌ preprocessing_word2vec.py bulunamadı!")
    st.error(f"Aranılan klasör: {METIN_SINIF_DIR}")
    st.stop()

# ===================== MODEL YOLLARI =====================

MODEL_A_PATH = MODELS_DIR / "mlp_model_Model_A_Temel_Optimize.h5"
MODEL_B_PATH = MODELS_DIR / "mlp_model_Model_B_Derin_Optimize.h5"
W2V_MODEL_PATH = MODELS_DIR / "word2vec_model.model"

# ===================== STOP WORDS =====================

if 'turkish' in stopwords.fileids():
    TURKISH_STOP_WORDS = set(stopwords.words('turkish'))
else:
    TURKISH_STOP_WORDS = {"bir", "ve", "ile", "de", "da", "bu", "o", "ki", "mi"}


# ===================== MODEL YÜKLEME VE SEÇİM =====================

@st.cache_resource
def load_and_select_best_model():
    """Model A ve Model B'yi yükler, en iyisini seçer."""

    # Word2Vec modelini yükle
    if not W2V_MODEL_PATH.exists():
        st.error(f"❌ Word2Vec modeli bulunamadı: {W2V_MODEL_PATH}")
        st.warning("➡ Önce 'preprocessing_word2vec.py' dosyasını çalıştırın!")
        return None, None, None, None, None

    try:
        w2v_model = Word2Vec.load(str(W2V_MODEL_PATH))
        st.success(f"✅ Word2Vec modeli yüklendi")
    except Exception as e:
        st.error(f"❌ Word2Vec yükleme hatası: {e}")
        return None, None, None, None, None

    # 2. MLP Modellerini Yükle
    model_a, model_b = None, None

    try:
        if MODEL_A_PATH.exists():
            model_a = load_model(str(MODEL_A_PATH), compile=False)
            st.success("✅ Model A yüklendi")
    except Exception as e:
        st.warning(f"⚠️ Model A yüklenemedi: {e}")
    try:
        if MODEL_B_PATH.exists():
            model_b = load_model(str(MODEL_B_PATH), compile=False)
            st.success("✅ Model B yüklendi")
    except Exception as e:
        st.warning(f"⚠️ Model B yüklenemedi: {e}")

    if model_a is None and model_b is None:
        st.error("❌ Hiçbir model yüklenemedi!")
        st.warning("➡ Önce 'mlp_sentiment_classifier.py' dosyasını çalıştırın!")
        return None, None, None, None, None

    # 3. En İyi Modeli Seç (MLP Çıktısına Göre)
    best_model = model_a if model_a is not None else model_b
    best_name = "Model A (Temel)" if best_model == model_a else "Model B (Derin)"  # Varsayılan

    best_name_file = MODELS_DIR / "best_model_name.txt"

    if best_name_file.exists():
        try:
            with open(best_name_file, "r", encoding="utf-8") as f:
                best_name_from_file = f.read().strip()

                # Dosyadan okunan modelin yüklü olup olmadığını kontrol et
                if "Model B" in best_name_from_file and model_b is not None:
                    best_model = model_b
                    best_name = best_name_from_file
                elif "Model A" in best_name_from_file and model_a is not None:
                    best_model = model_a
                    best_name = best_name_from_file
                # else: best_model, varsayılan atamada kalır
        except Exception as e:
            st.warning(f"⚠️ best_model_name.txt okunamadı: {e}. Varsayılan model kullanılıyor.")



    # 4. Label Encoder hazırla
    le = LabelEncoder()
    le.fit(["Pozitif", "Negatif", "Nötr"])

    # 5. Karşılaştırma Sonuçlarını Oku (Metrikler)
    comparison_results_path = METIN_SINIF_DIR / "en_iyi_model.csv"
    acc_a, acc_b = 0, 0
    if comparison_results_path.exists():
        try:
            df_comp = pd.read_csv(comparison_results_path)
            model_a_row = df_comp[df_comp['Model'] == 'Model A (Temel)']
            model_b_row = df_comp[df_comp['Model'] == 'Model B (Derin)']
            if not model_a_row.empty:
                acc_a = float(model_a_row['Doğruluk'].iloc[0])
            if not model_b_row.empty:
                acc_b = float(model_b_row['Doğruluk'].iloc[0])

        except Exception as e:
            st.warning(f"⚠️ Karşılaştırma CSV okunamadı: {e}")
            acc_a, acc_b = 0, 0

    st.info(f"🎯 Seçilen Model: **{best_name}**")

    # 6. Tüm 5 Değeri Döndür
    return best_model, w2v_model, le, acc_a, acc_b

# ===================== DUYGU ANALİZİ =====================

def apply_real_sentiment(df, model, w2v_model, label_encoder):
    """DataFrame'e duygu tahmini ekler."""

    if df.empty:
        return df

    tokenized = [preprocess_text(text) for text in df['Yorum_Metni']]

    # Filtreleme (Boş token listelerini ve None'ları kaldırır)
    valid_indices = [i for i, tokens in enumerate(tokenized) if tokens is not None and len(tokens) > 0]
    if len(valid_indices) == 0:
        # Hiç geçerli yorum yoksa
        df['Duygu_Durumu'] = 'Nötr'
        return df

        # Sadece geçerli tokenlar için vektör oluştur

    valid_tokens = [tokenized[i] for i in valid_indices]

    X = np.stack([document_vector_hybrid(w2v_model, t) for t in valid_tokens])

    preds = model.predict(X, verbose=0)
    labels = label_encoder.inverse_transform(np.argmax(preds, axis=1))

    df['Duygu_Durumu'] = 'Nötr'

    # Geçerli olan tahminleri yerleştir
    for idx, label in zip(valid_indices, labels):
        df.at[df.index[idx], 'Duygu_Durumu'] = label

    return df


# ===================== STREAMLIT ARAYÜZ =====================

st.set_page_config(
    page_title="YouTube Duygu Analizi",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #FF0000;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🎥 YouTube Yorumlarının Duygu Analizi</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Word2Vec + MLP ile Türkçe Duygu Sınıflandırması</p>', unsafe_allow_html=True)

# Sidebar - Model Bilgileri
with st.sidebar:
    st.header("📊 Model Bilgileri")

    result = load_and_select_best_model()

    BEST_MODEL, W2V_MODEL, LABEL_ENCODER, ACC_A, ACC_B = result

    if BEST_MODEL is None:
        st.error("❌ Model yüklenemedi!")
        st.stop()

    if ACC_A > 0 or ACC_B > 0:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Model A", f"{ACC_A:.2%}")
        with col2:
            st.metric("Model B", f"{ACC_B:.2%}")


# Session state
if 'comments_df' not in st.session_state:
    st.session_state.comments_df = pd.DataFrame()
if 'video_meta' not in st.session_state:
    st.session_state.video_meta = {}


# ===================== VERİ ÇEKME =====================

def fetch_and_process_video(url_or_id):
    """YouTube'dan veri çeker ve duygu analizi yapar."""

    st.session_state.comments_df = pd.DataFrame()

    vid = get_video_id_from_url(url_or_id)
    if not vid:
        st.error("❌ Geçerli bir video URL veya ID giriniz.")
        return

    with st.spinner("⏳ Yorumlar çekiliyor..."):
        try:
            meta, df = get_video_comments(vid)
        except Exception as e:
            st.error(f"❌ Hata: {e}")
            return

    with st.spinner("🔮 Duygu analizi yapılıyor..."):
        df = apply_real_sentiment(df, BEST_MODEL, W2V_MODEL, LABEL_ENCODER)

    st.session_state.video_meta = meta
    st.session_state.comments_df = df
    st.success("✔️ Veri çekildi ve duygu analizi tamamlandı!")


# ===================== ARAMA KUTUSU =====================

st.markdown("---")

col1, col2 = st.columns([4, 1])

with col1:
    query = st.text_input(
        "🔍 YouTube Video URL veya ID girin:",
        placeholder="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        help="Örnek: https://www.youtube.com/watch?v=VIDEO_ID veya sadece VIDEO_ID"
    )

with col2:
    st.write("")  # Boşluk
    st.write("")  # Boşluk
    analyze_btn = st.button("🔎 Analiz Et", use_container_width=True, type="primary")

if analyze_btn:
    if query:
        fetch_and_process_video(query)
    else:
        st.warning("⚠️ Lütfen bir URL veya ID girin!")

# ===================== SONUÇLARI GÖSTER =====================

if not st.session_state.comments_df.empty:

    meta = st.session_state.video_meta
    df = st.session_state.comments_df

    # Video Bilgileri
    st.markdown("---")
    st.subheader("📌 Video Bilgileri")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("📹 Video", meta.get('title', '-')[:20] + "...")
    with col2:
        st.metric("👤 Kanal", meta.get('channel_title', '-'))
    with col3:
        st.metric("👀 Görüntülenme", f"{int(meta.get('view_count', 0)):,}")
    with col4:
        st.metric("👍 Beğeni", f"{int(meta.get('like_count', 0)):,}")

    # Duygu Dağılımı
    st.markdown("---")
    st.subheader("📊 Duygu Analizi Sonuçları")

    total = len(df)
    pos = len(df[df['Duygu_Durumu'] == 'Pozitif'])
    neg = len(df[df['Duygu_Durumu'] == 'Negatif'])
    neu = len(df[df['Duygu_Durumu'] == 'Nötr'])

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("📊 Toplam Yorum", total)
    with col2:
        st.metric("😊 Pozitif", pos, f"{pos / total * 100:.1f}%")
    with col3:
        st.metric("😢 Negatif", neg, f"{neg / total * 100:.1f}%")
    with col4:
        st.metric("😐 Nötr", neu, f"{neu / total * 100:.1f}%")

    # Grafik - Duygu Dağılımı
    col1, col2 = st.columns(2)

    with col1:
        fig_pie = px.pie(
            values=[pos, neg, neu],
            names=['Pozitif', 'Negatif', 'Nötr'],
            title='Duygu Dağılımı (Pasta Grafik)',
            color_discrete_map={
                'Pozitif': '#28a745',
                'Negatif': '#dc3545',
                'Nötr': '#ffc107'
            },
            hole=0.3
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        fig_bar = px.bar(
            x=['Pozitif', 'Negatif', 'Nötr'],
            y=[pos, neg, neu],
            title='Duygu Sayıları (Bar Grafik)',
            labels={'x': 'Duygu', 'y': 'Yorum Sayısı'},
            color=['Pozitif', 'Negatif', 'Nötr'],
            color_discrete_map={
                'Pozitif': '#28a745',
                'Negatif': '#dc3545',
                'Nötr': '#ffc107'
            }
        )
        fig_bar.update_layout(showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    # Yorumlar Tablosu
    st.markdown("---")
    st.subheader("📝 Yorumlar ve Tahminler")

    # Filtre
    col1, col2 = st.columns([3, 1])
    with col1:
        filter_option = st.selectbox(
            "Duygu Filtreleme:",
            ["Tümü", "Pozitif", "Negatif", "Nötr"]
        )
    with col2:
        st.write("")
        st.write("")
        show_count = st.number_input("Gösterilecek yorum:", 5, len(df), 20)

    if filter_option != "Tümü":
        filtered_df = df[df['Duygu_Durumu'] == filter_option]
    else:
        filtered_df = df


    # Renk kodlaması için stil
    def color_sentiment(val):
        if val == 'Pozitif':
            return 'background-color: #d4edda; color: #155724'
        elif val == 'Negatif':
            return 'background-color: #f8d7da; color: #721c24'
        else:
            return 'background-color: #fff3cd; color: #856404'


    display_df = filtered_df.head(show_count)[
        ['Yorum_Metni', 'Yorum_Kullanici_Adi', 'Yorum_Like_Sayisi', 'Duygu_Durumu']
    ]

    display_df.index = np.arange(1, len(display_df) + 1)

    # İndeks adını "Sıra No" olarak değiştir
    display_df.index.name = "Sıra No"

    styled_df = display_df.style.applymap(
        color_sentiment, subset=['Duygu_Durumu']
    )

    st.dataframe(styled_df, use_container_width=True, height=400)

    # İstatistikler
    st.markdown("---")
    st.subheader("📈 İstatistikler")

    col1, col2, col3 = st.columns(3)

    with col1:
        avg_likes = df['Yorum_Like_Sayisi'].mean()
        st.metric("Ortalama Beğeni", f"{avg_likes:.1f}")

    with col2:
        most_liked = df.loc[df['Yorum_Like_Sayisi'].idxmax()]
        st.metric("En Çok Beğenilen", f"{most_liked['Yorum_Like_Sayisi']} 👍")
        st.caption(f"Kullanıcı: {most_liked['Yorum_Kullanici_Adi']}")

    with col3:
        sentiment_dist = df['Duygu_Durumu'].value_counts()
        dominant = sentiment_dist.idxmax()
        st.metric("Baskın Duygu", dominant)
        st.caption(f"{sentiment_dist[dominant]} yorum ({sentiment_dist[dominant] / total * 100:.1f}%)")

    # CSV Kaydetme
    st.markdown("---")
    st.subheader("💾 Veriyi Kaydet")

    col1, col2 = st.columns(2)

    with col1:
        save_path = SAVE_DIR / f"{meta.get('id', 'video')}_yorumlar.csv"
        if st.button("💾 Sunucuya CSV Olarak Kaydet", use_container_width=True):
            df.to_csv(save_path, index=False, encoding='utf-8')
            st.success(f"✅ CSV kaydedildi: {save_path.name}")

    with col2:
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Bilgisayarıma CSV İndir",
            data=csv_data,
            file_name=f"{meta.get('id', 'video')}_yorumlar.csv",
            mime="text/csv",
            use_container_width=True
        )

else:
    # Başlangıç ekranı
    st.info("👆 Yukarıdaki arama kutusuna bir YouTube video URL'si girin ve analiz edin!")

