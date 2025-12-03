import pandas as pd
import os
from googleapiclient.discovery import build
from dotenv import load_dotenv
from pathlib import Path

# ===================== KLASÖR YAPISI =====================


CURRENT_DIR = Path(__file__).resolve().parent  # VeriToplamaVeArayuz/
BASE_DIR = CURRENT_DIR.parent  # DuyguAnalizi/

env_path = BASE_DIR / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
    print(f"✅ .env dosyası yüklendi: {env_path}")
else:
    load_dotenv()  # Varsayılan yolu dene
    print("⚠️ .env dosyası bulunamadı, varsayılan yol deneniyor...")

# ===================== API AYARLARI =====================

# API anahtarını oku
API_KEY = os.getenv("YOUTUBE_API_KEY")

YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"

# ===================== FONKSİYONLAR =====================

def get_video_id_from_url(url_or_id):
    """ URL'den veya doğrudan girilen metinden Video ID'sini ayrıştırır."""

    url_or_id = url_or_id.strip()

    # Standart YouTube URL
    if "v=" in url_or_id:
        return url_or_id.split("v=")[-1].split("&")[0]

    # Kısa YouTube URL
    elif "youtu.be/" in url_or_id:
        return url_or_id.split("youtu.be/")[-1].split("?")[0]

    # Doğrudan ID
    return url_or_id


def get_video_comments(video_id):
    """Belirtilen video ID'sine ait yorumları ve meta verilerini çeker."""

    # API anahtarı kontrolü
    if not API_KEY:
        raise ValueError(
            "❌ YOUTUBE_API_KEY ortam değişkeni bulunamadı!\n"
            f"Lütfen .env dosyanızı kontrol edin: {BASE_DIR / '.env'}\n"
            "İçeriği şu şekilde olmalı:\n"
            "YOUTUBE_API_KEY=AIzaSyXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
        )

    # YouTube API istemcisini oluştur
    youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=API_KEY)

    # ==================== 1. VİDEO META VERİLERİNİ ÇEKME ====================
    try:
        video_response = youtube.videos().list(
            part='snippet,statistics',
            id=video_id
        ).execute()
    except Exception as e:
        raise Exception(f"❌ Video Meta Verisi Çekilemedi. API Hatası: {e}")

    if not video_response.get('items'):
        raise Exception("❌ Video ID'si bulunamadı veya geçersiz.")

    item = video_response['items'][0]

    # Video bilgilerini sakla
    video_info = {
        'id': video_id,
        'title': item['snippet']['title'],
        'published_at': item['snippet']['publishedAt'],
        'channel_title': item['snippet']['channelTitle'],
        'view_count': item['statistics'].get('viewCount', 0),
        'like_count': item['statistics'].get('likeCount', 0),
        'dislike_count': item['statistics'].get('dislikeCount', 0),  # YouTube dislike sayısını göstermiyor olduğu için bu kısım genellikle 0 olur
    }

    print(f"\n📹 Video: {video_info['title']}")
    print(f"📺 Kanal: {video_info['channel_title']}")
    print(f"👀 Görüntülenme: {video_info['view_count']}")

    # ==================== 2. YORUMLARI ÇEKME ====================

    comments = []
    next_page_token = None
    MAX_COMMENTS_LIMIT = 50  # Maksimum 50 yorum çek

    print(f"\n⏳ Yorumlar çekiliyor... (Maksimum {MAX_COMMENTS_LIMIT} yorum)")

    while len(comments) < MAX_COMMENTS_LIMIT:

        # Bu iterasyonda kaç yorum çekileceğini hesapla
        comments_to_fetch = min(75, MAX_COMMENTS_LIMIT - len(comments))

        try:
            results = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                textFormat="plainText",
                maxResults=comments_to_fetch,
                pageToken=next_page_token
            ).execute()

            # Yorumları işle
            for item in results["items"]:
                comment_snippet = item["snippet"]["topLevelComment"]["snippet"]

                comments.append({
                    'Video_Title': video_info['title'],
                    'Yayıncı_Kullanıcı_Adı': video_info['channel_title'],
                    'Video_Yayın_Tarihi': video_info['published_at'].split('T')[0],
                    'Video_Like_Sayısı': video_info['like_count'],
                    'Video_Dislike_Sayısı': video_info['dislike_count'],
                    'Yorum_Metni': comment_snippet["textDisplay"],
                    'Yorum_Tarihi': comment_snippet["publishedAt"].split('T')[0],
                    'Yorum_Kullanici_Adi': comment_snippet["authorDisplayName"],
                    'Yorum_Like_Sayisi': comment_snippet["likeCount"],
                    'Yorum_Dislike_Sayisi': comment_snippet.get("viewerRating", "none"),
                    'Yorum_Reply_Sayisi': item["snippet"]["totalReplyCount"],
                    'Duygu_Durumu': 'Nötr (Analiz Edilecek)',  # Başlangıç değeri
                })

            # Sonraki sayfa kontrolü
            next_page_token = results.get("nextPageToken")

            # Eğer sonraki sayfa yoksa veya sonuç yoksa dur
            if not next_page_token or len(results["items"]) == 0:
                break

        except Exception as e:
            print(f"⚠️ API Hatası (Durduruldu): {e}")
            break

    print(f"✅ {len(comments)} yorum çekildi.")

    return video_info, pd.DataFrame(comments)

# ===================== TEST =====================

if __name__ == "__main__":
    """
    python data_fetcher.py ile çalıştırılır.
    """

    print("\n" + "=" * 70)
    print("DATA_FETCHER TEST MODU")
    print("=" * 70)

    # Test için örnek video ID'leri
    test_videos = [
        "dQw4w9WgXcQ",
    ]

    test_video_id = test_videos[0]

    try:
        print(f"\n🔄 Test: Video yorumları çekiliyor...")
        print(f"📌 Video ID: {test_video_id}")

        meta, df = get_video_comments(test_video_id)

        print(f"\n{'=' * 70}")
        print("SONUÇLAR")
        print(f"{'=' * 70}")
        print(f"\n📹 Video: {meta['title']}")
        print(f"📺 Kanal: {meta['channel_title']}")
        print(f"👀 Görüntülenme: {meta['view_count']}")
        print(f"💬 Çekilen yorum sayısı: {len(df)}")

        if not df.empty:
            print(f"\n{'=' * 70}")
            print("İLK 3 YORUM:")
            print(f"{'=' * 70}")
            for idx, row in df.head(3).iterrows():
                print(f"\n{idx + 1}. {row['Yorum_Kullanici_Adi']}")
                print(f"   💬 {row['Yorum_Metni'][:100]}...")
                print(f"   👍 {row['Yorum_Like_Sayisi']} beğeni")

        # CSV'ye kaydet
        save_path = CURRENT_DIR / "exported_csv" / f"test_{test_video_id}_yorumlar.csv"
        save_path.parent.mkdir(exist_ok=True)
        df.to_csv(save_path, index=False, encoding='utf-8')
        print(f"\n✅ Test CSV kaydedildi: {save_path}")

    except Exception as e:
        print(f"\n❌ Test başarısız: {e}")
        print("\n💡 Kontrol listesi:")
        print("   1. .env dosyası var mı?")
        print("   2. YOUTUBE_API_KEY doğru mu?")
        print("   3. İnternet bağlantınız çalışıyor mu?")