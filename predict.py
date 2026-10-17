import os
import torch
from PIL import Image, ImageOps
import torchvision.transforms as transforms
from model import MNIST_MLP

def preprocess_custom_image(image_path):
    """
    Kendi çektiğimiz fotoğrafı MNIST standardına dönüştürür:
    - Gri tonlamaya çevirir
    - Rengi ters çevirir (Beyaz kağıt -> Siyah zemin, Siyah mürekkep -> Beyaz çizgi)
    - 28x28 boyutuna getirir
    - Tensor ve Normalize uygular
    """
    img = Image.open(image_path).convert('L')
    
    # 1. Invert: MNIST'te zemin siyah (0), rakam beyaz (255)'dir.
    img = ImageOps.invert(img)
    
    # 2. Kontrastı artırmak için basit bir eşikleme/otokontrast
    img = ImageOps.autocontrast(img, cutoff=2)
    
    # 3. 28x28 boyutuna yeniden boyutlandırma
    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    
    # İşlenmiş hali kaydetmek isterseniz kontrol amaçlı saklayın
    processed_dir = "processed_numbers"
    os.makedirs(processed_dir, exist_ok=True)
    img.save(os.path.join(processed_dir, os.path.basename(image_path)))

    # 4. Model transformasyonu (Eğitimdekiyle birebir aynı)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    tensor_img = transform(img).unsqueeze(0) # Batch boyutu ekle: (1, 1, 28, 28)
    return tensor_img

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Eğitilmiş modeli yükle
    model = MNIST_MLP()
    model.load_state_dict(torch.load("mnist_mlp_weights.pth", map_location=device))
    model.to(device)
    model.eval()

    # Öğrenci numaranızın rakamlarının bulunduğu klasör
    # Örneğin: my_numbers/ klasöründe digit_0.jpg, digit_1.jpg ... gibi dosyalar olmalı
    custom_images_dir = "my_numbers"
    
    if not os.path.exists(custom_images_dir):
        print(f"Lütfen '{custom_images_dir}' klasörü oluşturup içine öğrenci numaranızın rakamlarını kırpıp yükleyin.")
        return

    image_files = sorted([f for f in os.listdir(custom_images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    print("\n---  KENDİ EL YAZIM TEST SONUÇLARI ---")
    with torch.no_grad():
        for filename in image_files:
            img_path = os.path.join(custom_images_dir, filename)
            input_tensor = preprocess_custom_image(img_path).to(device)
            
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item() * 100

            print(f"Dosya: {filename:<15} | Model Tahmini: {predicted_class} | Güven: %{confidence:.2f}")

if __name__ == "__main__":
    main()