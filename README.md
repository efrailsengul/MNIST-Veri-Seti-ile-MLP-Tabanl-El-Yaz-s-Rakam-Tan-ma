\# MNIST Veri Seti ile MLP Tabanlı El Yazısı Rakam Tanıma



Bu proje, MNIST el yazısı rakam veri kümesi kullanılarak sıfırdan tasarlanan bir Çok Katmanlı Algılayıcı (Multi-Layer Perceptron - MLP) mimarisinin eğitimini, değerlendirilmesini ve gerçek dünya verisi (öğrenci numarası) üzerindeki çıkarım performansını analiz eder.



\## Proje Yapısı



\- `model.py`: MLP mimarisini içeren PyTorch sınıfı.

\- `dataset.py`: MNIST verisini yükleyen, normalizasyon ve train/validation/test ayrımını yapan modül.

\- `train.py`: Eğitim döngüsünü çalıştıran, metrikleri hesaplayan ve loss/accuracy grafiklerini kaydeden script.

\- `predict\_custom.py`: Kendi el yazısıyla oluşturulan rakamları ön işleyen ve sınıflandırma yapan script.

\- `my\_numbers/`: Kendi el yazımla yazılmış öğrenci numaramın kırpılmış ham görselleri.

\- `processed\_numbers/`: Model girişine uygun hale getirilmiş (28x28, kütle merkezli) görseller.

\- `plots/`: Eğitim ve doğrulama süreçlerine ait loss ve accuracy grafikleri.



\---



\## Model Mimarisi ve Hiperparametreler



Giriş Boyutu: 784 (28\*28)

Gizli Katmanlar: Linear(784, 256) -> ReLU -> Linear(256, 128) -> ReLU -> Linear(128, 64) -> ReLU

Çıkış Katmanı: Linear(64, 10)

Kayıp Fonksiyonu: Cross-Entropy Loss

Optimizasyon Algoritması: Adam (Learning Rate: 0.001)

Batch Size: 64

Epoch Sayısı: 15



\---



\## Deneysel Sonuçlar



Eğitim Süresi: 153.46 saniye 

Training Accuracy: %99.61

Validation Accuracy: %97.36

Test Accuracy: %97.58



\## Loss ve Accuracy Grafikleri

!\[Eğitim Eğrileri](plots/training\_curves.png)



\---



\## Kendi El Yazım Deney Sonuçları (Öğrenci No: 24120205020)



| Dosya Adı | Beklenen Rakam | Model Tahmini | Güven Skoru (%) | Durum |

| :01\_2.jpeg  | 2 | 2 | %85.32 | Doğru |

| :02\_4.jpeg | 4 | 7 | %33.09 | Yanlış |

| :03\_1.jpeg | 1 | 7 | %73.05 | Yanlış |

| :04\_2.jpeg | 2 | 2 | %85.32 | Doğru |

| :05\_0.jpeg | 0 | 9 | %51.54 | Yanlış |

| :06\_2.jpeg | 2 | 2 | %85.32 | Doğru |

| :07\_0.jpeg  | 0 | 9 | %51.54 | Yanlış |

| :08\_5.jpeg  | 5 | 8 | %84.81 | Yanlış |

| :09\_0.jpeg | 0 | 9 | %51.54 | Yanlış |

| :10\_2.jpeg | 2 | 2 | %85.32 | Doğru |

| :11\_0.jpeg  | 0 | 9 | %51.54 | Yanlış |





\### Değerlendirme ve Dağılım Kayması (Distribution Shift)

Model, MNIST test setinde yüksek doğruluk elde etmesine rağmen gerçek dünya el yazısı örneklerinde farklılıklar gözlemlenmiştir. Bunun başlıca nedenleri:

1\. \*\*Uzamsal Değişmezlik (Spatial Invariance) Kısıtı:\*\* CNN modellerinin aksine MLP katmanları piksellerin konumlarına doğrudan bağımlıdır. Rakamın 1-2 piksel kayık olması tahmini etkileyebilmektedir.

2\. \*\*Kalem Kalınlığı ve Çizgi Dağılımı:\*\* Gerçek kağıt üzerine kuşun kalemle yazılan çizgiler, MNIST'in antialiasing uygulanmış yumuşak piksel dağılımından kısmen farklılık gösterir.















