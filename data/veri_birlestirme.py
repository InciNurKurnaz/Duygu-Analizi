from pathlib import Path
import pandas as pd


# ===================== KLASÖR YAPISI =====================

DATA_DIR = Path(r"C:\DuyguAnalizi\data")  # Veri klasörünüzün yolu

TWITTER_TRAIN = DATA_DIR / "train_tweets_set.csv"
TWITTER_TEST = DATA_DIR / "test_tweets_set.csv"
STORE_REVIEWS = DATA_DIR / "magaza_yorumlari_duygu_analizi.csv"  # Senin dosya adın

OUTPUT_TRAIN = DATA_DIR / "train_set.csv"
OUTPUT_TEST = DATA_DIR / "test_set.csv"


# ===================== VERİ OKUMA =====================

def read_twitter_csv(filepath):
    """Twitter verilerini oku (;) ile ayrılmış"""
    data = []
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.rsplit(';', 1)
            if len(parts) == 2:
                text, sentiment = parts[0].strip(), parts[1].strip().lower()
                if text and sentiment:
                    data.append([text, sentiment])
    return pd.DataFrame(data, columns=['text', 'sentiment'])


def read_store_csv(filepath):
    """Mağaza yorumlarını oku (,) ile ayrılmış"""
    data = []
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Virgül ile ayır ama son virgül sentiment için
            parts = line.rsplit(',', 1)
            if len(parts) == 2:
                text, sentiment = parts[0].strip(), parts[1].strip().lower()
                if text and sentiment:
                    data.append([text, sentiment])
    return pd.DataFrame(data, columns=['text', 'sentiment'])


# ===================== ETİKET STANDARTLAŞTIRMA =====================

def standardize_labels(df):
    """Tüm etiketleri standart formata çevir"""
    label_map = {
        # Pozitif varyasyonlar
        'olumlu': 'pozitif', 'pozitif': 'pozitif', 'positive': 'pozitif',
        'iyi': 'pozitif', 'güzel': 'pozitif', '5': 'pozitif', '4': 'pozitif',

        # Negatif varyasyonlar
        'olumsuz': 'negatif', 'negatif': 'negatif', 'negative': 'negatif',
        'kötü': 'negatif', 'berbat': 'negatif', '1': 'negatif', '2': 'negatif',

        # Nötr varyasyonlar
        'tarafsız': 'notr', 'nötr': 'notr', 'notr': 'notr', 'neutral': 'notr',
        'orta': 'notr', 'fena değil': 'notr', '3': 'notr'
    }

    df['sentiment'] = df['sentiment'].map(label_map)
    df.dropna(subset=['sentiment'], inplace=True)
    return df


# ===================== ANA FONKSİYON =====================

def merge_datasets():
    """Tüm veri setlerini birleştir"""

    print("=" * 70)
    print("VERİ SETLERİNİ BİRLEŞTİRME")
    print("=" * 70)

    # 1. Twitter verilerini yükle
    print("\n📂 Twitter eğitim seti yükleniyor...")
    df_twitter_train = read_twitter_csv(TWITTER_TRAIN)
    print(f"✅ {len(df_twitter_train)} satır yüklendi")

    print("\n📂 Twitter test seti yükleniyor...")
    df_twitter_test = read_twitter_csv(TWITTER_TEST)
    print(f"✅ {len(df_twitter_test)} satır yüklendi")

    # 2. Mağaza yorumlarını yükle
    if STORE_REVIEWS.exists():
        print("\n📂 Mağaza yorumları yükleniyor...")
        df_store = read_store_csv(STORE_REVIEWS)
        print(f"✅ {len(df_store)} satır yüklendi")
    else:
        print(f"\n⚠️ Mağaza yorumları bulunamadı: {STORE_REVIEWS}")
        print("💡 Sadece Twitter verileri kullanılacak")
        df_store = pd.DataFrame(columns=['text', 'sentiment'])

    # 3. Etiketleri standartlaştır
    print("\n🔧 Etiketler standartlaştırılıyor...")
    df_twitter_train = standardize_labels(df_twitter_train)
    df_twitter_test = standardize_labels(df_twitter_test)
    if not df_store.empty:
        df_store = standardize_labels(df_store)

    # 4. Mağaza verilerini %80-20 böl
    if not df_store.empty:
        from sklearn.model_selection import train_test_split
        store_train, store_test = train_test_split(
            df_store, test_size=0.2, random_state=42,
            stratify=df_store['sentiment']
        )
        print(f"   Mağaza → Eğitim: {len(store_train)}, Test: {len(store_test)}")
    else:
        store_train = pd.DataFrame(columns=['text', 'sentiment'])
        store_test = pd.DataFrame(columns=['text', 'sentiment'])

    # 5. Birleştir
    combined_train = pd.concat([df_twitter_train, store_train], ignore_index=True)
    combined_test = pd.concat([df_twitter_test, store_test], ignore_index=True)

    # 6. Temizlik
    combined_train.drop_duplicates(subset=['text'], inplace=True)
    combined_test.drop_duplicates(subset=['text'], inplace=True)

    # 7. Sınıf dağılımı
    print("\n📊 Birleştirilmiş Eğitim Seti Dağılımı:")
    print(combined_train['sentiment'].value_counts())

    print("\n📊 Birleştirilmiş Test Seti Dağılımı:")
    print(combined_test['sentiment'].value_counts())

    # 8. Kaydet
    combined_train.to_csv(OUTPUT_TRAIN, sep=';', index=False, header=False)
    combined_test.to_csv(OUTPUT_TEST, sep=';', index=False, header=False)

    print(f"\n✅ Birleştirilmiş eğitim seti: {OUTPUT_TRAIN}")
    print(f"   Toplam: {len(combined_train)} satır")

    print(f"\n✅ Birleştirilmiş test seti: {OUTPUT_TEST}")
    print(f"   Toplam: {len(combined_test)} satır")

    print("\n💡 Şimdi preprocessing_word2vec.py'yi çalıştırın!")
    print("   (combined_train.csv ve combined_test.csv kullanılacak)")


if __name__ == "__main__":
    merge_datasets()