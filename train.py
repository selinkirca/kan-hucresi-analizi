import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from tqdm import tqdm

# ==========================================
# 1. HİPERPARAMETRELER VE AYARLAR
# ==========================================
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
NUM_CLASSES = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ekran görüntüsündeki klasör yapısına göre yol tanımı
# train.py dosyası ile 'dataset1' yan yana olmalıdır.
DATASET_PATH = "./dataset1" 

print(f"Çalışma Cihazı: {DEVICE}")

# ==========================================
# 2. VERİ ÖN İŞLEME VE DİNAMİK BÖLME
# ==========================================
# Görselleri yüklerken ortak bir transform uyguluyoruz
base_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(), # Eğitim başarısını artırmak için veri artırma
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Tüm veri setini yüklüyoruz
full_dataset = datasets.ImageFolder(root=DATASET_PATH, transform=base_transforms)

# Veriyi %80 Train, %20 Test olacak şekilde matematiksel olarak bölüyoruz
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size

# random_split verileri rastgele dağıtır, böylece her sınıftan dengeli bölünme sağlanır
train_dataset, test_dataset = random_split(
    full_dataset, 
    [train_size, test_size], 
    generator=torch.Generator().manual_seed(42) # Her çalıştırmada aynı rastgelelik olsun diye seed sabitlendi
)

# DataLoader'ların oluşturulması
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

print(f"Toplam Görsel Sayısı: {len(full_dataset)}")
print(f"Eğitim (Train) için ayrılan: {len(train_dataset)}")
print(f"Test için ayrılan: {len(test_dataset)}")
print(f"Bulunan Sınıflar: {full_dataset.classes}")

# ==========================================
# 3. MODEL TANIMLAMA FONKSİYONU
# ==========================================
def build_model(model_name):
    if model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
        for param in model.parameters():
            param.requires_grad = False
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
        
    elif model_name == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.DEFAULT
        model = models.mobilenet_v2(weights=weights)
        for param in model.parameters():
            param.requires_grad = False
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, NUM_CLASSES)
    else:
        raise ValueError("Geçersiz model adı!")
        
    return model.to(DEVICE)

# ==========================================
# 4. EĞİTİM VE TEST FONKSİYONU
# ==========================================
def train_and_validate(model, model_name):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    
    best_acc = 0.0
    print(f"\n--- {model_name.upper()} Eğitimi Başlıyor ---")
    
    for epoch in range(EPOCHS):
        #--- TRAIN ---
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for images, labels in train_loop:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
            train_loop.set_postfix(loss=loss.item())
            
        epoch_train_loss = running_loss / len(train_dataset)
        epoch_train_acc = (correct_train / total_train) * 100
        
        #--- TEST ---
        model.eval()
        running_test_loss = 0.0
        correct_test = 0
        total_test = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                running_test_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()
                
        epoch_test_loss = running_test_loss / len(test_dataset)
        epoch_test_acc = (correct_test / total_test) * 100
        
        print(f"Epoch {epoch+1} -> Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}%")
        print(f"Epoch {epoch+1} -> Test Loss: {epoch_test_loss:.4f} | Test Acc: {epoch_test_acc:.2f}%")
        
        # En iyi skoru kaydediyoruz
        if epoch_test_acc > best_acc:
            best_acc = epoch_test_acc
            torch.save(model.state_dict(), f"best_{model_name}.pth")
            print(f"==> En iyi {model_name} modeli kaydedildi! Başarı: {best_acc:.2f}%")

# ==========================================
# 5. ÇALIŞTIRMA
# ==========================================
if __name__ == "__main__":
    # ResNet50 Eğitimi
    resnet_model = build_model("resnet50")
    train_and_validate(resnet_model, "resnet50")
    
    # MobileNetV2 Eğitimi
    mobilenet_model = build_model("mobilenet_v2")
    train_and_validate(mobilenet_model, "mobilenet_v2")