import matplotlib.pyplot as plt
import os
from matplotlib.lines import Line2D # Line2D'yi doğrudan içe aktaralım

print("--- YAPAY SİNİR AĞI ÇİZDİRİLİYOR (KERAS YAPISINA UYGUN) ---")

# --- TOPOLOJİ SABİTLERİ (Keras Fonksiyonundan Çıkarıldı) ---
# Keras fonksiyonunuzdaki varsayılan topolojiyi temsil eder:
# model.add(Dense(neurons[0], ...))
# model.add(Dense(neurons[1], ...))
# model.add(Dense(neurons[2], ...))
# model.add(Dense(3, activation='softmax', ...))

# Örnek varsayılan değerler
input_size = 64 # Temsili bir giriş boyutu (input_dim)
hidden_sizes_list = [128, 64, 32] # Örnek 'neurons' listesi (neurons[0], neurons[1], neurons[2])
output_size = 3 # Çıkış katmanı her zaman 3 sınıflıdır

# Tüm katman boyutları listesi
layers = [input_size] + hidden_sizes_list + [output_size]

print(f"Gerçek Katman Yapısı (Temsili Girişli): {layers}")
print("Görsel oluşturuluyor...")


# --- ÇİZİM FONKSİYONU ---
def draw_neural_net(ax, left, right, bottom, top, layer_sizes):
    """Verilen katman boyutlarına göre bir sinir ağı mimarisini çizer."""
    v_spacing = (top - bottom) / float(max(layer_sizes))
    h_spacing = (right - left) / float(len(layer_sizes) - 1)

    # Düğümleri Çiz
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing * (layer_size - 1) / 2. + (top + bottom) / 2.
        for m in range(layer_size):
            circle = plt.Circle((n * h_spacing + left, layer_top - m * v_spacing), v_spacing / 4.,
                                color='w', ec='k', zorder=4)
            ax.add_artist(circle)

            # Etiketler
            # Sadece en üstteki nöronlara yazı yaz
            if m == 0:
                y_text_ust = layer_top + v_spacing
                y_text_alt = layer_top - (layer_size) * v_spacing - v_spacing

                if n == 0: # Giriş Katmanı
                    plt.text(n * h_spacing + left, y_text_ust, "**Giriş\nKatmanı**", ha='center', fontsize=12,
                             fontweight='bold')
                    plt.text(n * h_spacing + left, y_text_alt,
                             f"{layer_sizes[n]}\nÖzellik (input_dim)", ha='center', color='blue')
                elif n == len(layer_sizes) - 1: # Çıkış Katmanı
                    plt.text(n * h_spacing + left, y_text_ust, "**Çıkış\nKatmanı**", ha='center', fontsize=12,
                             fontweight='bold', color='red')
                    plt.text(n * h_spacing + left, y_text_alt,
                             f"{layer_sizes[n]}\nSınıf (Softmax)", ha='center', color='red')
                else: # Gizli Katmanlar
                    plt.text(n * h_spacing + left, y_text_ust, f"**Gizli\nKatman {n}**", ha='center',
                             fontsize=10)
                    plt.text(n * h_spacing + left, y_text_alt,
                             f"{layer_sizes[n]}\nNöron (ReLU)", ha='center')

    # Bağlantıları Çiz
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing * (layer_size_a - 1) / 2. + (top + bottom) / 2.
        layer_top_b = v_spacing * (layer_size_b - 1) / 2. + (top + bottom) / 2.
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                line = Line2D([n * h_spacing + left, (n + 1) * h_spacing + left],
                              [layer_top_a - m * v_spacing, layer_top_b - o * v_spacing], c='gray', alpha=0.3)
                ax.add_artist(line)


# --- ÇİZİMİ BAŞLAT ---
fig = plt.figure(figsize=(14, 8))
ax = fig.gca()
ax.axis('off')

# Çizimde karışıklığı önlemek için temsili nöron sayıları kullanalım
# Ancak katman sayısını (4 katman: Giriş + 3 Gizli + Çıkış) koruyalım.
# layers = [input_size, 128, 64, 32, 3] -> 5 katman
if len(layers) > 5:
    # Çok fazla nöron varsa gösterimi küçültelim
    temsili_yapi = [6, 5, 4, 3, 3]
else:
    # Katman sayıları zaten azsa direkt gerçek sayıları kullanalım (veya daha temsili)
    temsili_yapi = [layers[0] if layers[0] < 10 else 8] + \
                   [size if size < 10 else 6 for size in layers[1:-1]] + \
                   [layers[-1]]

draw_neural_net(ax, .05, .95, .1, .9, temsili_yapi)

plt.title(f"Keras MLP Mimarisi ({len(layers)} Katman)", fontsize=16, fontweight='bold')

# Resmi Kaydet (Örnek olarak mevcut dizine kaydediyoruz)
# Bu_dosyanin_yeri yerine sadece . (mevcut dizin) kullanmak daha güvenli.
kayit_adi = "keras_sinir_agi_mimarisi.png"
plt.savefig(kayit_adi, dpi=300, bbox_inches='tight')
print(f"✅ Şekil kaydedildi: {os.path.abspath(kayit_adi)}")

plt.show()

print("\n--- ÖNEMLİ NOT ---")
print("Bu görsel, Keras modelinizin (Dense Katmanları) nöron sayılarını temsili olarak gösterir.")
print("**Dropout** ve **Batch Normalization** gibi ek katmanlar, genellikle mimari şemasında düğüm olarak gösterilmez, ancak bağlantıları etkiler.")