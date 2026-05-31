import os
import shutil
import random
import urllib.request
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# Yolların Tanımlanması
BASE_DIR = r"C:\Users\selin\OneDrive\Masaüstü\blood"
TRAIN_DATASET_DIR = os.path.join(BASE_DIR, "dataset1")
GATEKEEPER_DIR = os.path.join(BASE_DIR, "gatekeeper_dataset")

INVALID_DIR = os.path.join(GATEKEEPER_DIR, "0_invalid")
VALID_DIR = os.path.join(GATEKEEPER_DIR, "1_valid")

# Klasörleri Güvenli Oluştur
os.makedirs(INVALID_DIR, exist_ok=True)
os.makedirs(VALID_DIR, exist_ok=True)

# ==============================================================================
# 🌍 1. KISIM: İNTERNETTEN RASTGELE ANOMALİ GÖRSEL ÇEKME (0_invalid)
# ==============================================================================
print("[INFO] İnternetten anomali test görselleri (arayüz, yüz, manzara) indiriliyor...")

urls = [
    "https://picsum.photos/300/300", 
    "https://images.unsplash.com/photo-1559839734-2b71ea197ec2?w=300", # Yüz
    "https://images.unsplash.com/photo-1454165804606-c3d57bc86b40?w=300", # Belge/Yazı
    "https://images.unsplash.com/photo-1516321318423-f06f85e504b3?w=300"  # Ekran/Şema
]

for i in range(40):
    try:
        target_url = random.choice(urls)
        if "picsum" in target_url:
            target_url = f"https://picsum.photos/300/300?random={i}"
            
        dest_path = os.path.join(INVALID_DIR, f"invalid_{i}.jpg")
        req = urllib.request.Request(target_url, headers={'User-Agent': 'Mozilla/5.5'})
        with urllib.request.urlopen(req, timeout=5) as response, open(dest_path, 'wb') as out_file:
            out_file.write(response.read())
    except Exception:
        continue

print(f"✅ [{len(os.listdir(INVALID_DIR))}] Adet Alakasız Görsel Hazırlandı.")

# ==============================================================================
# 🩸 2. KISIM: KAN DATASETİNDEN GERÇEK HÜCRE FOTOĞRAFLARI KOPYALAMA (1_valid)
# ==============================================================================
print("[INFO] Kan hücreleri datasetinden dengeli örnekler kopyalanıyor...")
siniflar = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']

hucre_sayac = 0
for sinif in siniflar:
    sinif_yolu = os.path.join(TRAIN_DATASET_DIR, sinif)
    if os.path.exists(sinif_yolu):
        dosyalar = [f for f in os.listdir(sinif_yolu) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        secilenler = random.sample(dosyalar, min(15, len(dosyalar)))
        
        for dosya in secilenler:
            kaynak = os.path.join(sinif_yolu, dosya)
            hedef = os.path.join(VALID_DIR, f"valid_{sinif}_{hucre_sayac}.jpg")
            shutil.copy(kaynak, hedef)
            hucre_sayac += 1

print(f"✅ [{len(os.listdir(VALID_DIR))}] Adet Gerçek Kan Hücresi Fotoğrafı Hazırlandı.")

# ==============================================================================
# 🧠 3. KISIM: KORUYUCU BINARY MODELİN EĞİTİLMESİ
# ==============================================================================
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = datagen.flow_from_directory(
    GATEKEEPER_DIR, target_size=(224, 224), batch_size=4, class_mode='binary', subset='training'
)
val_gen = datagen.flow_from_directory(
    GATEKEEPER_DIR, target_size=(224, 224), batch_size=4, class_mode='binary', subset='validation'
)

base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False 

gate_model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1, activation='sigmoid')
])

gate_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
gate_model.fit(train_gen, validation_data=val_gen, epochs=5, verbose=1)

gate_model.save(os.path.join(BASE_DIR, "best_blood_gatekeeper.keras"))
print("\n🎯 [BAŞARILI] 'best_blood_gatekeeper.keras' koruma modeli üretildi!")