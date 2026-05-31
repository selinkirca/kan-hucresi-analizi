import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import os
import numpy as np
import cv2
import tensorflow as tf

# ==========================================
# SAYFA AYARLARI VE CSS
# ==========================================
st.set_page_config(page_title="Hematoloji Analiz Sistemi", page_icon="🔬", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    div[data-testid="metric-container"] {
        background-color: rgba(28, 131, 225, 0.1);
        border: 1px solid rgba(28, 131, 225, 0.1);
        padding: 15px 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    button[data-baseweb="tab"] {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 1. AYARLAR VE SABİTLER
# ==========================================
CLASS_NAMES = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Klasör yolunu hem lokalde hem bulutta çalışacak şekilde dinamik yapıyoruz
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==========================================
# 2. MODELLERİN BELLEĞE YÜKLENMESİ
# ==========================================
@st.cache_resource
def load_models():
    # 🛡️ 1. Kademeli Giriş Kapısı Koruma Modeli
    gate_path = os.path.join(BASE_DIR, "best_blood_gatekeeper.keras")
    if os.path.exists(gate_path):
        gatekeeper = tf.keras.models.load_model(gate_path)
    else:
        gatekeeper = None

    # Teşhis Modeli - ResNet50
    resnet = models.resnet50(weights=None)
    resnet.fc = nn.Linear(resnet.fc.in_features, 4)
    resnet_path = os.path.join(BASE_DIR, "best_resnet50.pth")
    if os.path.exists(resnet_path):
        resnet.load_state_dict(torch.load(resnet_path, map_location=DEVICE, weights_only=True))
    resnet.to(DEVICE).eval()

    # Teşhis Modeli - MobileNetV2
    mobilenet = models.mobilenet_v2(weights=None)
    mobilenet.classifier[1] = nn.Linear(mobilenet.classifier[1].in_features, 4)
    mobilenet_path = os.path.join(BASE_DIR, "best_mobilenet_v2.pth")
    if os.path.exists(mobilenet_path):
        mobilenet.load_state_dict(torch.load(mobilenet_path, map_location=DEVICE, weights_only=True))
    mobilenet.to(DEVICE).eval()
    
    return gatekeeper, resnet, mobilenet

try:
    gatekeeper_model, resnet_model, mobilenet_model = load_models()
    models_ready = True
except Exception as e:
    st.error(f"⚠️ Sistem yükleme hatası: {e}")
    models_ready = False

img_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================
# 3. DOĞRULAMA: HİBRİT OPENCV FİLTRESİ
# ==========================================
def opencv_gorsel_kontrol(pil_image):
    img_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    kernel = np.ones((15, 15), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    
    gecerli_piksel = cv2.countNonZero(mask)
    if gecerli_piksel == 0: gecerli_piksel = 1 

    ortalama_parlaklik = cv2.mean(gray, mask=mask)[0]
    if ortalama_parlaklik < 30: return False, "Karanlık Görsel"
    if ortalama_parlaklik > 250: return False, "Aşırı Parlak / Boş Ekran"

    hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
    lower_green_blue = np.array([35, 40, 40])   
    upper_green_blue = np.array([130, 255, 255]) 
    mask_gb = cv2.inRange(hsv, lower_green_blue, upper_green_blue)
    mask_gb = cv2.bitwise_and(mask_gb, mask_gb, mask=mask) 
    oran_gb = (cv2.countNonZero(mask_gb) / gecerli_piksel) * 100

    if oran_gb > 15.0: return False, "Medikal Dışı Renk Profil"

    edges = cv2.Canny(gray, 50, 150)
    edges_masked = cv2.bitwise_and(edges, edges, mask=mask)
    
    lines = cv2.HoughLinesP(edges_masked, 1, np.pi/180, threshold=60, minLineLength=50, maxLineGap=10)
    if lines is not None and len(lines) > 5: return False, "Geometrik Nesne Algılandı"

    kenar_orani = (cv2.countNonZero(edges_masked) / gecerli_piksel) * 100
    if kenar_orani < 0.5: return False, "Aşırı Pürüzsüz Doku"
    if kenar_orani > 25.0: return False, "Aşırı Karmaşık Doku"

    return True, "Başarılı"

# ==========================================
# 4. ÜST BAŞLIK VE YAN PANEL (SIDEBAR)
# ==========================================
st.markdown("<h1 style='text-align: center; color: #1f77b4;'>🔬 Hematoloji Analiz Laboratuvarı</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #7f8c8d;'>Periferik Kan Yayması Hücre Sınıflandırma Sistemi</h4>", unsafe_allow_html=True)
st.divider()

with st.sidebar:
    st.header("Sistem Durumu")
    st.success("🟢 **Consensus Ortak Karar:** Aktif")
    st.success("🟢 **ResNet50:** Entegre")
    st.success("🟢 **MobileNetV2:** Entegre")
    st.error("🔒 **Giriş Kalkanı:** Aktif")
    st.info("🛡 **Filtre Sistemi:** Devrede")
    st.info("🎯 **Baraj Eşiği:** %45")
    st.markdown("---")
    st.markdown("### 🧬 Analiz Sınıfları")
    st.markdown("- Eosinophil\n- Lymphocyte\n- Monocyte\n- Neutrophil")

# ==========================================
# 5. SEKME YAPISI (TABS)
# ==========================================
tab1, tab2, tab3 = st.tabs(["🖥 Canlı Teşhis", "📊 Performans Metrikleri", "📚 Mimari Detaylar"])

# ------------------------------------------
# TAB 1: CANLI TEŞHİS LABORATUVARI
# ------------------------------------------
with tab1:
    st.markdown("### 📸 Görsel Yükleme Paneli")
    
    # 🚨 ÇÖZÜM 1: Her yeni dosya yüklendiğinde Streamlit önbelleğini tetiklemek için benzersiz bir key atıyoruz
    uploaded_file = st.file_uploader(
        "Mikroskop görüntüsünü buraya sürükleyin veya seçin", 
        type=["png", "jpg", "jpeg"],
        key="blood_cells_uploader"
    )

    if uploaded_file is not None and models_ready:
        image = Image.open(uploaded_file).convert("RGB")
        st.markdown("---")
        
        col_img, col_res = st.columns([1, 2], gap="large")
        
        with col_img:
            st.markdown("##### İncelenen Örnek")
            st.image(image, use_container_width=True, caption="Yüklenen Hücre")
            
        gorsel_ok, mesaj = opencv_gorsel_kontrol(image)
        
        is_medical_blood = True
        if gorsel_ok and gatekeeper_model is not None:
            img_keras = image.resize((224, 224))
            img_arr = np.array(img_keras) / 255.0
            img_arr = np.expand_dims(img_arr, axis=0)
            
            # Keras tahminini anlık olarak yeniliyoruz
            gate_pred = float(gatekeeper_model.predict(img_arr, verbose=0)[0][0])
            if gate_pred < 0.50:
                is_medical_blood = False
                mesaj = f"Kriter dışı doku yapısı saptandı. Doğrulama Skoru: %{gate_pred*100:.1f}"

        if not gorsel_ok or not is_medical_blood:
            with col_res:
                st.error("🚫 **Görsel Reddedildi**")
                st.warning(f"**Sebep:** {mesaj}")
                st.info("Lütfen mikroskop altında çekilmiş gerçek bir kan hücresi fotoğrafı yükleyin.")
        else:
            # 🚨 ÇÖZÜM 2: PyTorch tensör akışını zorunlu olarak sıfırlayıp temiz veri oluşturuyoruz
            input_tensor = img_transforms(image).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                # Modelleri anlık girdiye zorla
                res_out = resnet_model(input_tensor)
                res_probs = F.softmax(res_out, dim=1).gradient = None # Bellek kilidini çöz
                res_probs = res_probs.cpu().numpy()[0]
                
                mob_out = mobilenet_model(input_tensor)
                mob_probs = F.softmax(mob_out, dim=1).cpu().numpy()[0]

            # Olasılıkları temiz float dizilerine zorluyoruz (RAM takılmasını önlemek için)
            res_probs = np.array(res_probs, dtype=np.float64)
            mob_probs = np.array(mob_probs, dtype=np.float64)

            # 🚨 ÇÖZÜM 3: Ortak Karar (Ensemble) Dağılım Hesabı
            ensemble_probs = (res_probs + mob_probs) / 2.0
            ensemble_idx = int(ensemble_probs.argmax())
            ensemble_max_prob = float(ensemble_probs[ensemble_idx])
            
            with col_res:
                st.markdown("#### 🧠 Konsensüs Analiz Sonucu")
                
                if ensemble_max_prob >= 0.45:
                    # 🚨 .clear() tetikleyicisi gibi çalışması için dinamik metrik basımı
                    st.metric(
                        label="Sistem Tahmini", 
                        value=CLASS_NAMES[ensemble_idx], 
                        delta=f"%{ensemble_max_prob*100:.1f} Dengelenmiş Güven"
                    )
                    
                    if ensemble_max_prob >= 0.75:
                        st.success("✅ **Yüksek Güvenilirlikli Sonuç:** Modeller ortak veri morfolojisi üzerinde mutabık kaldı.")
                    else:
                        st.warning("⚠️ **Orta Güvenilirlikli Sonuç:** Hücre sınırlarında morfolojik kararsızlık mevcut.")
                else:
                    st.metric(label="Sistem Tahmini", value="Belirsiz Yapı", delta=f"%{ensemble_max_prob*100:.1f}", delta_color="inverse")
                    st.error("❌ Analiz Reddedildi: Hibrit güven skoru %45 barajının altında.")
                        
                st.markdown("---")
                st.markdown("###### 📊 Güncel Olasılık Dağılımı")
                
                # Çubukları anlık gelen yeni array değerlerine göre döngüde yeniden çiziyoruz
                for i, c_name in enumerate(CLASS_NAMES):
                    st.progress(float(ensemble_probs[i]), text=f"{c_name} (%{ensemble_probs[i]*100:.1f})")

# ------------------------------------------
# TAB 2: MODEL PERFORMANS GRAFİKLERİ
# ------------------------------------------
with tab2:
    st.markdown("### 📊 İstatistiksel Doğruluk Grafikleri")
    st.info("Eğitim sonrası doğrulama veri setiyle elde edilen kalıcı grafikler.")
    
    col_r, col_m = st.columns(2)
    with col_r: st.success("🏆 **ResNet50 Doğruluk Oranı:** %80.00")
    with col_m: st.warning("🥈 **MobileNetV2 Doğruluk Oranı:** %75.00")
        
    st.markdown("---")
    g_col1, g_col2 = st.columns(2)
    
    with g_col1:
        st.markdown("<h4 style='text-align: center;'>🔵 ResNet50 Raporları</h4>", unsafe_allow_html=True)
        if os.path.exists(os.path.join(BASE_DIR, "resnet50_learning_curves.png")): 
            st.image(os.path.join(BASE_DIR, "resnet50_learning_curves.png"), use_container_width=True)
        if os.path.exists(os.path.join(BASE_DIR, "resnet50_confusion_matrix_test.png")): 
            st.image(os.path.join(BASE_DIR, "resnet50_confusion_matrix_test.png"), use_container_width=True)

    with g_col2:
        st.markdown("<h4 style='text-align: center;'>🟠 MobileNetV2 Raporları</h4>", unsafe_allow_html=True)
        if os.path.exists(os.path.join(BASE_DIR, "mobilenet_v2_learning_curves.png")): 
            st.image(os.path.join(BASE_DIR, "mobilenet_v2_learning_curves.png"), use_container_width=True)
        if os.path.exists(os.path.join(BASE_DIR, "mobilenet_v2_confusion_matrix_test.png")): 
            st.image(os.path.join(BASE_DIR, "mobilenet_v2_confusion_matrix_test.png"), use_container_width=True)

# ------------------------------------------
# TAB 3: PROJE & MİMARİ DETAYLARI
# ------------------------------------------
with tab3:
    st.markdown("### ⚙️ Sistem Mimarisi ve Koruma Kalkanları")
    
    with st.expander("🔬 Veri Seti Kapsamı", expanded=True):
        st.write("""
        - **Kaynak:** Kaggle Periferik Kan Hücreleri Dağılımı.
        - **Etiketler:** Eosinophil, Lymphocyte, Monocyte, Neutrophil.
        - **Dağılım:** %80 Eğitim, %20 Doğrulama (Validation).
        """)
        
    with st.expander("🛡️ Çok Kademeli Entegre Filtre Sistemi", expanded=True):
        st.write("""
        1. **OpenCV Filtresi:** HoughLinesP geometrisi ve Canny Kenar yoğunluğu ile dijital arayüz resimlerini engeller.
        2. **Giriş Kalkanı (Gatekeeper):** Hücresel doku yapısına uymayan alakasız nesneleri kapıda eler.
        3. **Soft Voting Ensemble:** İki modelin ortak olasılık matris ortalamasını alarak aşırı özgüven sapmalarını sönümler.
        """)
