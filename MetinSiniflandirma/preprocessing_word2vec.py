import pandas as pd
import numpy as np
from pathlib import Path
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.data import find as nltk_find
from nltk import download as nltk_download
from nltk.tokenize import word_tokenize
from zemberek import TurkishMorphology
import re
import os
import sys

# TensorFlow uyarılarını kısıtla (Burada gerekmez ama temiz kod için kalsın)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ===================== ZEMBEREK İLK BAŞLATMA =====================

try:
    LEMMATIZER = TurkishMorphology.create_with_defaults()
    # print("✅ Zemberek Morfoloji Başlatıldı.")
except Exception as e:
    print(f"❌ HATA: Zemberek başlatılamadı: {e}")
    sys.exit(1)

# ===================== NLTK KAYNAKLARI =====================

try:
    nltk_find('corpora/stopwords')
except:
    nltk_download('stopwords')

try:
    nltk_find('tokenizers/punkt')
except:
    nltk_download('punkt')

# ===================== STOP WORDS =====================

if "turkish" in stopwords.fileids():
    TURKISH_STOP_WORDS = set(stopwords.words("turkish"))
else:
    TURKISH_STOP_WORDS = set([
        "bir", "ve", "ile", "de", "da", "bu", "o", "ki", "ama", "fakat",
        "gibi", "ancak", "yani", "hem", "ya", "mı", "mi", "mü", "mu"
    ])

# ===================== KLASÖR YAPISI =====================

CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent
DATA_DIR = BASE_DIR / "data"
VECTORS_OUTPUT_DIR = DATA_DIR / "vectors"
MODELS_DIR = CURRENT_DIR / "models"
WORD2VEC_MODEL_PATH = MODELS_DIR / "word2vec_model.model"

TRAIN_FILE = DATA_DIR / "train_set.csv"
TEST_FILE = DATA_DIR / "test_set.csv"

TRAIN_VEC_OUTPUT_FILE = VECTORS_OUTPUT_DIR / "X_train_vectors.csv"
TEST_VEC_OUTPUT_FILE = VECTORS_OUTPUT_DIR / "X_test_vectors.csv"

MODELS_DIR.mkdir(exist_ok=True)
VECTORS_OUTPUT_DIR.mkdir(exist_ok=True)


# ===================== METİN TEMİZLEME (OPTİMİZE) =====================

def preprocess_text(text):
    """Metni temizler, tokenleştirir ve köklerini (lemma) bulur (Hızlı Versiyon)."""
    if pd.isna(text):
        return []

    text = str(text)

    # Temizlik Adımları
    text = re.sub(r'http\S+|www\S+|https\S+', ' ', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', ' ', text)
    text = text.replace("\n", " ").replace("\r", " ").replace("<br />", " ").lower()
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'[^a-zığüşöç\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    tokens = word_tokenize(text, language='turkish')
    lemmatized_tokens = []

    for w in tokens:
        # HIZLI KÖK BULMA: disambiguate çağrısı kaldırıldı.
        analysis_list = LEMMATIZER.analyze(w)

        try:
            # Kontrol: Analiz yoksa kelimenin kendisini kullan
            if not analysis_list or not analysis_list[0].analysis_results:
                primary_root = w
            else:
                # En olası (ilk) analizin kökünü al
                primary_root = analysis_list[0].best.get_lemma()

        except Exception:
            primary_root = w

        lemmatized_tokens.append(primary_root)

    # STOP WORD ve UZUNLUK filtrelemesini KÖK bulunmuş tokenler üzerinde uygula
    lemmatized_tokens = [w for w in lemmatized_tokens if w not in TURKISH_STOP_WORDS and len(w) > 2]

    return lemmatized_tokens


# ===================== WORD2VEC MODEL YÖNETİMİ =====================

def load_or_create_word2vec(vector_size=100, window=5, min_count=2, sg=1):
    """Kayıtlı Word2Vec modelini yükler veya sıfırdan oluşturur."""

    if WORD2VEC_MODEL_PATH.exists():
        try:
            model = Word2Vec.load(str(WORD2VEC_MODEL_PATH))
            print(f"✅ Word2Vec modeli yüklendi. Vektör Boyutu: {model.vector_size}")
            return model
        except Exception as e:
            print(f"❌ HATA: Model yüklenirken hata oluştu ({e}). Sıfırdan oluşturuluyor...")

    print("🛠️ Word2Vec modeli sıfırdan oluşturuluyor.")
    return Word2Vec(
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=sg,
        workers=4
    )


def update_and_save_word2vec(model, new_sentences, epochs=30):
    """Word2Vec modelini yeni verilerle günceller ve kaydeder."""

    initial_words = len(model.wv)
    # build_vocab ile yeni kelimeleri ekle (update=True)
    model.build_vocab(new_sentences, update=True)
    new_words = len(model.wv) - initial_words

    if len(new_sentences) == 0 or model.corpus_total_words == 0:
        print("⚠️ Uyarı: Yeni veri seti boş veya geçerli kelime içermiyor. Eğitim atlandı.")
        return model

    print(f"📌 Word2Vec: Kelime dağarcığı güncellendi (+{new_words} yeni kelime). Toplam: {len(model.wv)}")

    # train ile modeli eğit
    print("📌 Word2Vec: Model güncelleniyor/eğitiliyor...")
    model.train(
        new_sentences,
        total_examples=model.corpus_count,
        epochs=epochs
    )

    model.save(str(WORD2VEC_MODEL_PATH))
    print(f"✅ Word2Vec modeli güncellendi ve kaydedildi: {WORD2VEC_MODEL_PATH}")
    return model


# ===================== HİBRİT VEKTÖR (MEAN + MAX) =====================

def document_vector_hybrid(model, tokens):
    """
    Kelime vektörlerinin hem MEAN hem de MAX değerlerini alır.
    Sonuç: 100 (mean) + 100 (max) = 200 boyutlu vektör
    """
    if not tokens:
        return np.zeros(model.vector_size * 2)

    wv = model.wv

    vectors = [wv[w] for w in tokens if w in wv]

    if not vectors:
        return np.zeros(model.vector_size * 2)

    mean_vec = np.mean(vectors, axis=0)
    max_vec = np.max(vectors, axis=0)

    return np.hstack([mean_vec, max_vec])


# ===================== CSV OKUMA FONKSİYONU (EKLENDİ) =====================

def safe_read_csv(filepath):
    """CSV dosyasını güvenli şekilde okur."""
    data = []

    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            header = next(f, None)

            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                # CSV'nin ayırıcısını otomatik belirle
                separator = ';' if ';' in line else (',' if ',' in line else None)

                if separator:
                    # Son sütun sentiment, geri kalanı metin
                    parts = line.rsplit(separator, 1)
                else:
                    continue

                if len(parts) == 2:
                    text = parts[0].strip()
                    sentiment = parts[1].strip().lower()

                    if text and sentiment:
                        data.append([text, sentiment])

        df = pd.DataFrame(data, columns=['CommentText', 'Sentiment'])
        return df

    except FileNotFoundError:
        print(f"❌ HATA: Dosya bulunamadı: {filepath}")
        return pd.DataFrame()
    except Exception as e:
        print(f"❌ HATA: CSV okuma hatası: {e}")
        return pd.DataFrame()


# ===================== ANA PIPELINE =====================

def run_word2vec_pipeline(train_path, test_path):
    """Word2Vec eğitim pipeline'ını çalıştırır ve vektörleri oluşturur."""

    print(f"\n{'=' * 70}")
    print(f"ADIM 1: CSV DOSYALARINI OKUMA")
    print(f"{'=' * 70}")

    df_train = safe_read_csv(train_path)
    df_test = safe_read_csv(test_path)

    if df_train.empty or df_test.empty:
        print("❌ HATA: CSV dosyaları boş veya yüklenemedi!")
        return None

    print(f"✅ Eğitim seti: {len(df_train)} satır, Test seti: {len(df_test)} satır")

    print(f"\n{'=' * 70}")
    print(f"ADIM 2: VERİ TEMİZLEME")
    print(f"{'=' * 70}")
    label_map = {
        'olumlu': 'pozitif', 'pozitif': 'pozitif', 'olumsuz': 'negatif', 'negatif': 'negatif',
        'tarafsız': 'notr', 'nötr': 'notr', 'notr': 'notr', 'neutral': 'notr'
    }
    df_train['Sentiment'] = df_train['Sentiment'].map(label_map)
    df_test['Sentiment'] = df_test['Sentiment'].map(label_map)
    df_train.dropna(subset=['Sentiment'], inplace=True)
    df_test.dropna(subset=['Sentiment'], inplace=True)
    for df in [df_train, df_test]:
        df["CommentText"] = (df["CommentText"].astype(str).str.replace("\n", " ", regex=False).str.replace("\r", " ",
                                                                                                           regex=False).str.replace(
            "<br />", " ", regex=False).str.strip())
        df.drop_duplicates(subset=["CommentText"], inplace=True)

    df_combined = pd.concat([df_train, df_test], ignore_index=True)

    # 3. Tokenleştirme
    print(f"\n{'=' * 70}")
    print(f"ADIM 3: TOKENLEŞTİRME & KÖK BULMA")
    print(f"{'=' * 70}")

    comments = df_combined["CommentText"].tolist()
    tokenized = [preprocess_text(t) for t in comments]
    tokenized = [t for t in tokenized if len(t) > 0]

    if len(tokenized) < 50:
        print("❌ HATA: Tokenize edilen cümle sayısı çok az!")
        return None

    print(f"✅ Tokenize cümle sayısı: {len(tokenized)}")

    # 4. Word2Vec Modeli (Yükle/Güncelle)
    print(f"\n{'=' * 70}")
    print(f"ADIM 4: WORD2VEC MODELİ YÖNETİMİ (YÜKLE/GÜNCELLE)")
    print(f"{'=' * 70}")

    model = load_or_create_word2vec(vector_size=100)
    model = update_and_save_word2vec(model, tokenized)

    if model is None:
        return None

    # 5. HİBRİT VEKTÖRLER (MEAN + MAX)
    print(f"\n{'=' * 70}")
    print(f"ADIM 5: HİBRİT VEKTÖR OLUŞTURMA (MEAN + MAX)")
    print(f"{'=' * 70}")

    print("📊 Eğitim seti vektörleri hesaplanıyor...")
    df_train["tokens"] = df_train["CommentText"].apply(preprocess_text)
    df_train["vector"] = df_train["tokens"].apply(lambda x: document_vector_hybrid(model, x))
    x_train = np.vstack(df_train["vector"].values)
    y_train = df_train["Sentiment"].values
    vector_train_df = pd.DataFrame(x_train)
    vector_train_df.columns = [f'v{i}' for i in range(x_train.shape[1])]
    vector_train_df["sentiment"] = y_train

    print("📊 Test seti vektörleri hesaplanıyor...")
    df_test["tokens"] = df_test["CommentText"].apply(preprocess_text)
    df_test["vector"] = df_test["tokens"].apply(lambda x: document_vector_hybrid(model, x))
    x_test = np.vstack(df_test["vector"].values)
    y_test = df_test["Sentiment"].values
    vector_test_df = pd.DataFrame(x_test)
    vector_test_df.columns = [f'v{i}' for i in range(x_test.shape[1])]
    vector_test_df["sentiment"] = y_test

    # 6. Kaydetme
    print(f"\n{'=' * 70}")
    print(f"ADIM 6: VEKTÖRLERİ KAYDETME")
    print(f"{'=' * 70}")

    vector_train_df.to_csv(TRAIN_VEC_OUTPUT_FILE, index=False)
    print(f"✅ Eğitim vektörleri kaydedildi: {TRAIN_VEC_OUTPUT_FILE}")
    vector_test_df.to_csv(TEST_VEC_OUTPUT_FILE, index=False)
    print(f"✅ Test vektörleri kaydedildi: {TEST_VEC_OUTPUT_FILE}")

    print("\n➡ Artık 'mlp_sentiment_classifier.py' dosyasını çalıştırabilirsiniz!")
    return model


# ===================== MAIN =====================

if __name__ == "__main__":
    print(f"\n{'#' * 70}")
    print(f"### WORD2VEC ÖN İŞLEME (SÜREKLİ ÖĞRENME) ###")
    print(f"{'#' * 70}\n")
    if not TRAIN_FILE.exists() or not TEST_FILE.exists():
        print("❌ HATA: Eğitim veya test dosyası bulunamadı. Lütfen dosya yollarını kontrol edin.")
        sys.exit(1)

    run_word2vec_pipeline(TRAIN_FILE, TEST_FILE)