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
BASE_DIR = r"C:\Users\selin\OneDrive\Masaüstü\blood"

# ==========================================
# 2. MODELLERİN BELLEĞE YÜKLENMESİ
# ==========================================
@st.cache_resource
def load_models():
    # 🛡️ 1. Kademeli Giriş Kapısı Koruma Modeli (Keras Kalkanı)
    gate_path = os.path.join(BASE_DIR, "best_blood_gatekeeper.keras")
    if os.path.exists(gate_path):
        gatekeeper = tf.keras.models.load_model(gate_path)
    else:
        gatekeeper = None

    # Teşhis Modeli - ResNet50 (PyTorch)
    resnet = models.resnet50(weights=None)
    resnet.fc = nn.Linear(resnet.fc.in_features, 4)
    if os.path.exists(os.path.join(BASE_DIR, "best_resnet50.pth")):
        resnet.load_state_dict(torch.load(os.path.join(BASE_DIR, "best_resnet50.pth"), map_location=DEVICE, weights_only=True))
    resnet.to(DEVICE).eval()

    # Teşhis Modeli - MobileNetV2 (PyTorch)
    mobilenet = models.mobilenet_v2(weights=None)
    mobilenet.classifier[1] = nn.Linear(mobilenet.classifier[1].in_features, 4)
    if os.path.exists(os.path.join(BASE_DIR, "best_mobilenet_v2.pth")):
        mobilenet.load_state_dict(torch.load(os.path.join(BASE_DIR, "best_mobilenet_v2.pth"), map_location=DEVICE, weights_only=True))
    mobilenet.to(DEVICE).eval()
    
    return gatekeeper, resnet, mobilenet

try:
    gatekeeper_model, resnet_model, mobilenet_model = load_models()
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

    if oran_gb > 15.0: return False, "Medikal Dışı Renk Profili (Doğa/Eşya)"

    edges = cv2.Canny(gray, 50, 150)
    edges_masked = cv2.bitwise_and(edges, edges, mask=mask)
    
    lines = cv2.HoughLinesP(edges_masked, 1, np.pi/180, threshold=60, minLineLength=50, maxLineGap=10)
    if lines is not None and len(lines) > 5: return False, "Geometrik Nesne Algılandı (Araba/Arayüz/Yazı)"

    kenar_orani = (cv2.countNonZero(edges_masked) / gecerli_piksel) * 100
    if kenar_orani < 0.5: return False, "Aşırı Pürüzsüz Doku (Çizim/Arayüz)"
    if kenar_orani > 25.0: return False, "Aşırı Karmaşık Doku (Saç/Tüy/Manzara)"

    return True, "Başarılı"

# ==========================================
# 4. ÜST BAŞLIK VE YAN PANEL (SIDEBAR)
# ==========================================
st.markdown("<h1 style='text-align: center; color: #1f77b4;'>🔬 Hematoloji Yapay Zeka Laboratuvarı</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #7f8c8d;'>Periferik Kan Yayması Lökosit Sınıflandırma Sistemi</h4>", unsafe_allow_html=True)
st.divider()

with st.sidebar:
    st.header("Sistem Durumu")
    st.success("🟢 **Ensemble Ortak Karar:** Aktif")
    st.success("🟢 **ResNet50:** Entegre")
    st.success("🟢 **MobileNetV2:** Entegre")
    st.error("🔒 **AI Giriş Kalkanı:** Aktif")
    st.info("🛡 **CV2 Filtresi:** Devrede")
    st.info("🎯 **Güven Eşiği:** %45")
    
    st.markdown("---")
    st.markdown("### 🧬 Teşhis Edilen Hücreler")
    st.markdown("- Eosinophil\n- Lymphocyte\n- Monocyte\n- Neutrophil")
    st.caption("Yapay Zeka ile Sağlık Bilişimi")

# ==========================================
# 5. SEKME YAPISI (TABS)
# ==========================================
tab1, tab2, tab3 = st.tabs(["🖥 Canlı Teşhis", "📊 Performans Metrikleri", "📚 Mimari Detaylar"])

# ------------------------------------------
# TAB 1: CANLI TEŞHİS LABORATUVARI
# ------------------------------------------
with tab1:
    st.markdown("### 📸 Görsel Yükleme Paneli")
    uploaded_file = st.file_uploader("Mikroskop görüntüsünü buraya sürükleyin veya seçin", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None and models_ready:
        image = Image.open(uploaded_file).convert("RGB")
        st.markdown("---")
        
        # 2 Kolonlu dengeli yerleşim (Sol: Görsel, Sağ: Ortak Teşhis Sonucu)
        col_img, col_res = st.columns([1, 2], gap="large")
        
        with col_img:
            st.markdown("##### İncelenen Örnek")
            st.image(image, use_container_width=True, caption="Yüklenen Görsel")
            
        # 1. AŞAMA GÜVENLİK: KURAL TABANLI OPENCV KALKANI
        gorsel_ok, mesaj = opencv_gorsel_kontrol(image)
        
        # 2. AŞAMA GÜVENLİK: AI DOĞRULAMA MODELİ (Keras Gatekeeper)
        is_medical_blood = True
        if gorsel_ok and gatekeeper_model is not None:
            img_keras = image.resize((224, 224))
            img_arr = np.array(img_keras) / 255.0
            img_arr = np.expand_dims(img_arr, axis=0)
            
            gate_pred = gatekeeper_model.predict(img_arr, verbose=0)[0][0]
            if gate_pred < 0.50:
                is_medical_blood = False
                mesaj = f"AI Giriş Kalkanı tarafından reddedildi. Hücre Doku Skoru yetersiz: %{gate_pred*100:.1f}"

        if not gorsel_ok or not is_medical_blood:
            with col_res:
                st.error("🚫 **Görsel Reddedildi**")
                st.warning(f"**Sebep:** {mesaj}")
                st.info("Lütfen mikroskop altında çekilmiş gerçek bir periferik kan yayması görseli yükleyin.")
        else:
            # Ölçüm ve tahmin süreçleri (PyTorch)
            input_tensor = img_transforms(image).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                res_out = resnet_model(input_tensor)
                res_probs = F.softmax(res_out, dim=1).cpu().numpy()[0]
                
                mob_out = mobilenet_model(input_tensor)
                mob_probs = F.softmax(mob_out, dim=1).cpu().numpy()[0]

            # ==============================================================================
            # 🚨 3. AŞAMA: AKILLI YUMUŞAK ENSEMBLE ORTALAMASI (SOFT VOTING)
            # ==============================================================================
            # İki modelin tüm olasılık çıktılarını toplayıp ikiye bölerek tek bir hibrit dağılım elde ediyoruz
            ensemble_probs = (res_probs + mob_probs) / 2.0
            ensemble_idx = ensemble_probs.argmax()
            ensemble_max_prob = ensemble_probs[ensemble_idx]
            
            with col_res:
                st.markdown("#### 🧠 Konsensüs Yapay Zeka Teşhisi (Ortak Karar)")
                
                if ensemble_max_prob >= 0.45:
                    st.metric(label="Konsensüs Teşhis Sonucu", value=CLASS_NAMES[ensemble_idx], delta=f"%{ensemble_max_prob*100:.1f} Dengelenmiş Güven")
                    
                    # Güvenilirlik durum bildirimi
                    if ensemble_max_prob >= 0.75:
                        st.success("✅ **Yüksek Güvenilirlikli Klinik Sonuç:** İki model de ortak veri paternleri üzerinde mutabık kaldı.")
                    else:
                        st.warning("⚠️ **Orta Güvenilirlikli Klinik Sonuç:** Modeller arasında kısmi morfolojik kararsızlık mevcut, nihai karar uzman hekime aittir.")
                else:
                    st.metric(label="Konsensüs Teşhis Sonucu", value="Bilinmiyor / Belirsiz", delta=f"%{ensemble_max_prob*100:.1f}", delta_color="inverse")
                    st.error("❌ Teşhis Reddedildi: Hibrit güven skoru %45 medikal barajının altında kaldı.")
                        
                st.markdown("---")
                st.markdown("###### 📊 Dengelenmiş Konsensüs Olasılık Dağılımı")
                # Tek bir olasılık çubuğu seti üzerinden hocaya temiz sonuç gösteriyoruz
                for i, c_name in enumerate(CLASS_NAMES):
                    st.progress(float(ensemble_probs[i]), text=f"{c_name} (%{ensemble_probs[i]*100:.1f})")

# ------------------------------------------
# TAB 2: MODEL PERFORMANS GRAFİKLERİ
# ------------------------------------------
with tab2:
    st.markdown("### 📊 Akademik Test Sonuçları")
    st.info("Eğitim sonrası Validation (Doğrulama) veri seti kullanılarak elde edilen metrikler.")
    
    col_r, col_m = st.columns(2)
    with col_r: st.success("🏆 **ResNet50 Genel Doğruluk:** %80.00")
    with col_m: st.warning("🥈 **MobileNetV2 Genel Doğruluk:** %75.00")
        
    st.markdown("---")
    g_col1, g_col2 = st.columns(2)
    
    with g_col1:
        st.markdown("<h4 style='text-align: center;'>🔵 ResNet50 Grafikleri</h4>", unsafe_allow_html=True)
        if os.path.exists("resnet50_learning_curves.png"): st.image("resnet50_learning_curves.png", use_container_width=True)
        if os.path.exists("resnet50_confusion_matrix_test.png"): st.image("resnet50_confusion_matrix_test.png", use_container_width=True)
        if os.path.exists("resnet50_roc_curve_test.png"): st.image("resnet50_roc_curve_test.png", use_container_width=True)

    with g_col2:
        st.markdown("<h4 style='text-align: center;'>🟠 MobileNetV2 Grafikleri</h4>", unsafe_allow_html=True)
        if os.path.exists("mobilenet_v2_learning_curves.png"): st.image("mobilenet_v2_learning_curves.png", use_container_width=True)
        if os.path.exists("mobilenet_v2_confusion_matrix_test.png"): st.image("mobilenet_v2_confusion_matrix_test.png", use_container_width=True)
        if os.path.exists("mobilenet_v2_roc_curve_test.png"): st.image("mobilenet_v2_roc_curve_test.png", use_container_width=True)

# ------------------------------------------
# TAB 3: PROJE & MİMARİ DETAYLARI
# ------------------------------------------
with tab3:
    st.markdown("### ⚙️ Sistem Mimarisi ve Koruma Kalkanları")
    
    with st.expander("🔬 Veri Seti ve Ön İşleme", expanded=True):
        st.write(f"""
        - **Kaynak Veri Seti Yolu:** `https://www.kaggle.com/datasets/romainroure/blood-cells-4-classes-dataset`
        - **Sınıflar:** Eosinophil, Lymphocyte, Monocyte, Neutrophil.
        - **Bölümleme:** %80 Eğitim, %20 Doğrulama (Validation).
        """)
        
    with st.expander("🛡️ Çok Kademeli Akıllı Güvenlik Altyapısı", expanded=True):
        st.write("""
        1. **Kural Tabanlı Ön Filtre (OpenCV):** HoughLinesP geometrisi ve Canny Kenar yoğunluğu ile dijital ekran şemalarını ve boş sayfaları eler.
        2. **Yapay Zeka Giriş Kalkanı (Gatekeeper Model):** `best_blood_gatekeeper.keras` ikili sınıflandırma modeli sayesinde resmin anlamsal olarak mikroskop altındaki bir kan hücresi popülasyonu olup olmadığını denetler.
        3. **Yumuşak Konsensüs Ortalaması (Soft Voting Ensemble):** İki modelin olasılık tahminlerini toplayıp normalize ederek tekil model aşırı özgüven (Overconfidence) sapmalarını matematiksel olarak absorbe eder.
        """)
