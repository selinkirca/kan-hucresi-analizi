import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import os
import numpy as np
import cv2

# ==========================================
# SAYFA AYARLARI VE CSS
# ==========================================
st.set_page_config(page_title="Hematoloji Yapay Zeka Laboratuvarı", page_icon="🔬", layout="wide", initial_sidebar_state="expanded")

# Arayüzü modernleştirmek için Özel CSS Kodları
st.markdown("""
    <style>
    /* Ana menü ve footer gizleme */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Üst başlık boşluğunu daraltma */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Metrik (Teşhis) kutularını kart şeklinde tasarlama */
    div[data-testid="metric-container"] {
        background-color: rgba(28, 131, 225, 0.1);
        border: 1px solid rgba(28, 131, 225, 0.1);
        padding: 15px 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Sekme (Tab) başlıklarını büyütme */
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

# ==========================================
# 2. MODELLERİN BELLEĞE YÜKLENMESİ
# ==========================================
@st.cache_resource
def load_models():
    resnet = models.resnet50(weights=None)
    resnet.fc = nn.Linear(resnet.fc.in_features, 4)
    if os.path.exists("best_resnet50.pth"):
        resnet.load_state_dict(torch.load("best_resnet50.pth", map_location=DEVICE, weights_only=True))
    resnet.to(DEVICE)
    resnet.eval()

    mobilenet = models.mobilenet_v2(weights=None)
    mobilenet.classifier[1] = nn.Linear(mobilenet.classifier[1].in_features, 4)
    if os.path.exists("best_mobilenet_v2.pth"):
        mobilenet.load_state_dict(torch.load("best_mobilenet_v2.pth", map_location=DEVICE, weights_only=True))
    mobilenet.to(DEVICE)
    mobilenet.eval()
    
    return resnet, mobilenet

try:
    resnet_model, mobilenet_model = load_models()
    models_ready = True
except Exception as e:
    st.error(f"⚠️ Model dosyaları yüklenirken hata oluştu: {e}")
    models_ready = False

img_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================
# 3. DOĞRULAMA: OPENCV FİLTRESİ
# ==========================================
def opencv_gorsel_kontrol(pil_image):
    img_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    kernel = np.ones((15, 15), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    
    gecerli_piksel = cv2.countNonZero(mask)
    if gecerli_piksel == 0:
        gecerli_piksel = 1 

    ortalama_parlaklik = cv2.mean(gray, mask=mask)[0]
    if ortalama_parlaklik < 30: 
        return False, "Karanlık Görsel"
    if ortalama_parlaklik > 250:
        return False, "Aşırı Parlak / Boş Ekran"

    hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
    lower_green_blue = np.array([35, 40, 40])   
    upper_green_blue = np.array([130, 255, 255]) 
    mask_gb = cv2.inRange(hsv, lower_green_blue, upper_green_blue)
    mask_gb = cv2.bitwise_and(mask_gb, mask_gb, mask=mask) 
    oran_gb = (cv2.countNonZero(mask_gb) / gecerli_piksel) * 100

    if oran_gb > 15.0: 
        return False, "Medikal Dışı Renk Profili (Doğa/Eşya)"

    edges = cv2.Canny(gray, 50, 150)
    edges_masked = cv2.bitwise_and(edges, edges, mask=mask)
    
    lines = cv2.HoughLinesP(edges_masked, 1, np.pi/180, threshold=60, minLineLength=50, maxLineGap=10)
    if lines is not None and len(lines) > 5: 
        return False, "Geometrik Nesne Algılandı (Araba/Arayüz/Yazı)"

    kenar_orani = (cv2.countNonZero(edges_masked) / gecerli_piksel) * 100
    if kenar_orani < 0.5:
        return False, "Aşırı Pürüzsüz Doku (Çizim/Arayüz)"
    if kenar_orani > 25.0: 
        return False, "Aşırı Karmaşık Doku (Saç/Tüy/Manzara)"

    return True, "Başarılı"

# ==========================================
# 4. ÜST BAŞLIK VE YAN PANEL (SIDEBAR)
# ==========================================
st.markdown("<h1 style='text-align: center; color: #1f77b4;'>🔬 Hematoloji Yapay Zeka Laboratuvarı</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #7f8c8d;'>Periferik Kan Yayması Lökosit Sınıflandırma Sistemi</h4>", unsafe_allow_html=True)
st.divider()

with st.sidebar:
    st.header("Sistem Durumu")
    st.success("🟢 **ResNet50:** Aktif")
    st.success("🟢 **MobileNetV2:** Aktif")
    st.info("🛡️ **CV2 Filtresi:** Devrede")
    st.info("🎯 **Güven Eşiği:** %45")
    
    st.markdown("---")
    st.markdown("### 🧬 Teşhis Edilen Hücreler")
    st.markdown("- Eosinophil\n- Lymphocyte\n- Monocyte\n- Neutrophil")
    st.caption("Yapay Zeka ile Sağlık Bilişimi")

# ==========================================
# 5. SEKME YAPISI (TABS)
# ==========================================
tab1, tab2, tab3 = st.tabs(["🖥️ Canlı Teşhis", "📊 Performans Metrikleri", "📚 Mimari Detaylar"])

# ------------------------------------------
# TAB 1: CANLI TEŞHİS LABORATUVARI
# ------------------------------------------
with tab1:
    st.markdown("### 📸 Görsel Yükleme Paneli")
    uploaded_file = st.file_uploader("Mikroskop görüntüsünü buraya sürükleyin veya seçin", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None and models_ready:
        image = Image.open(uploaded_file).convert("RGB")
        st.markdown("---")
        
        # 3 Kolonlu modern görünüm
        col_img, col_res, col_mob = st.columns([1, 1.2, 1.2], gap="large")
        
        with col_img:
            st.markdown("##### İncelenen Örnek")
            st.image(image, use_container_width=True, caption="Yüklenen Görsel")
            
        # 1. AŞAMA GÜVENLİK
        gorsel_ok, mesaj = opencv_gorsel_kontrol(image)
        
        if not gorsel_ok:
            with col_res:
                st.error("🚫 **Görsel Reddedildi**")
                st.warning(f"**Sebep:** {mesaj}")
                st.info("Lütfen sadece medikal kan yayması yükleyin.")
        else:
            input_tensor = img_transforms(image).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                res_out = resnet_model(input_tensor)
                res_probs = F.softmax(res_out, dim=1).cpu().numpy()[0]
                res_idx = res_probs.argmax()
                res_max_prob = res_probs[res_idx]
                
                mob_out = mobilenet_model(input_tensor)
                mob_probs = F.softmax(mob_out, dim=1).cpu().numpy()[0]
                mob_idx = mob_probs.argmax()
                mob_max_prob = mob_probs[mob_idx]

            # 2. AŞAMA GÜVENLİK
            if res_max_prob < 0.45 and mob_max_prob < 0.45:
                with col_res:
                    st.error("⚠️ **Düşük Güven Skoru**")
                    st.warning("Yapay zeka modelleri bu görseli hiçbir kan hücresine %45'ten fazla benzetemedi.")
            else:
                with col_res:
                    st.markdown("#### 🧠 ResNet50 Teşhisi")
                    if res_max_prob >= 0.45:
                        st.metric(label="Birincil Tahmin", value=CLASS_NAMES[res_idx], delta=f"%{res_max_prob*100:.1f} Emin")
                        if res_max_prob >= 0.80:
                            st.success("Yüksek Güvenilirlikli Sonuç")
                        else:
                            st.warning("Orta Güvenilirlikli Sonuç")
                    else:
                        st.metric(label="Tahmin", value="Bilinmiyor", delta=f"%{res_max_prob*100:.1f}", delta_color="inverse")
                        
                    st.markdown("###### Olasılık Dağılımı")
                    for i, c_name in enumerate(CLASS_NAMES):
                        st.progress(float(res_probs[i]), text=f"{c_name} (%{res_probs[i]*100:.1f})")

                with col_mob:
                    st.markdown("#### ⚡ MobileNetV2 Teşhisi")
                    if mob_max_prob >= 0.45:
                        st.metric(label="İkincil Tahmin", value=CLASS_NAMES[mob_idx], delta=f"%{mob_max_prob*100:.1f} Emin")
                        if mob_max_prob >= 0.80:
                            st.success("Yüksek Güvenilirlikli Sonuç")
                        else:
                            st.warning("Orta Güvenilirlikli Sonuç")
                    else:
                        st.metric(label="Tahmin", value="Bilinmiyor", delta=f"%{mob_max_prob*100:.1f}", delta_color="inverse")
                        
                    st.markdown("###### Olasılık Dağılımı")
                    for i, c_name in enumerate(CLASS_NAMES):
                        st.progress(float(mob_probs[i]), text=f"{c_name} (%{mob_probs[i]*100:.1f})")

# ------------------------------------------
# TAB 2: MODEL PERFORMANS GRAFİKLERİ
# ------------------------------------------
with tab2:
    st.markdown("### 📊 Akademik Test Sonuçları")
    st.info("Eğitim sonrası Validation (Doğrulama) veri seti kullanılarak elde edilen metrikler.")
    
    col_r, col_m = st.columns(2)
    with col_r:
        st.success("🏆 **ResNet50 Genel Doğruluk:** %80.00")
    with col_m:
        st.warning("🥈 **MobileNetV2 Genel Doğruluk:** %75.00")
        
    st.markdown("---")
    
    g_col1, g_col2 = st.columns(2)
    
    with g_col1:
        st.markdown("<h4 style='text-align: center;'>🔵 ResNet50 Grafikleri</h4>", unsafe_allow_html=True)
        if os.path.exists("resnet50_learning_curves.png"):
            st.image("resnet50_learning_curves.png", use_container_width=True)
        if os.path.exists("resnet50_confusion_matrix_test.png"):
            st.image("resnet50_confusion_matrix_test.png", use_container_width=True)
        if os.path.exists("resnet50_roc_curve_test.png"):
            st.image("resnet50_roc_curve_test.png", use_container_width=True)

    with g_col2:
        st.markdown("<h4 style='text-align: center;'>🟠 MobileNetV2 Grafikleri</h4>", unsafe_allow_html=True)
        if os.path.exists("mobilenet_v2_learning_curves.png"):
            st.image("mobilenet_v2_learning_curves.png", use_container_width=True)
        if os.path.exists("mobilenet_v2_confusion_matrix_test.png"):
            st.image("mobilenet_v2_confusion_matrix_test.png", use_container_width=True)
        if os.path.exists("mobilenet_v2_roc_curve_test.png"):
            st.image("mobilenet_v2_roc_curve_test.png", use_container_width=True)

# ------------------------------------------
# TAB 3: PROJE & MİMARİ DETAYLARI
# ------------------------------------------
with tab3:
    st.markdown("### ⚙️ Sistem Mimarisi ve Koruma Kalkanları")
    
    with st.expander("🔬 Veri Seti ve Ön İşleme", expanded=True):
        st.write("""
        - **Sınıflar:** Eosinophil, Lymphocyte, Monocyte, Neutrophil.
        - **Bölümleme:** Dinamik olarak %80 Eğitim, %20 Test.
        - **Veri Artırma (Data Augmentation):** Modelin ezberlemesini önlemek için yatay döndürme ve 15 derecelik açılarla döndürme uygulandı.
        """)
        
    with st.expander("🛡️ OpenCV ve Görüntü İşleme Güvenliği", expanded=True):
        st.write("""
        Kullanıcıların sisteme araba, manzara, yüz veya uygulama arayüzü yüklemesini engellemek için 4 katmanlı biyolojik filtre yazılmıştır:
        1. **Dinamik Siyah Maskeleme:** Augmentation kaynaklı siyah rotasyon boşlukları tespit edilip analiz dışı bırakılır.
        2. **HSV Renk Analizi:** Sadece medikal boya renklerine izin verilir.
        3. **HoughLinesP (Geometri):** Görseldeki düz çizgileri (arayüz, harf, araba) tespit edip bloklar.
        4. **Canny Edge (Doku):** Mikroskobik hücre dokusuna uymayan aşırı pürüzsüz veya aşırı pürüzlü fotoğrafları engeller.
        """)
        
    with st.expander("🧠 Derin Öğrenme Modelleri", expanded=True):
        st.write("""
        - **ResNet50:** Katmanlar arası atlama bağlantıları sayesinde derin ağlardaki gradyan kaybını önler. (%80 Doğruluk)
        - **MobileNetV2:** Tersine çevrilmiş artık bloklar (Inverted Residuals) kullanarak mobil ve kısıtlı cihazlar için optimize edilmiştir. (%75 Doğruluk)
        - **Yapay Zeka Barajı:** Modellerin tahmini %45 güven skorunun altındaysa sistem sonucu "Bilinmiyor" olarak reddeder.
        """)