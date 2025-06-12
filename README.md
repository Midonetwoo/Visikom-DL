# Visikom-DL

Nama: Muhammad Dirga Apriliansyah

NIM: 2209106050

Proyek ini bertujuan untuk membangun dan mengevaluasi model machine learning menggunakan pendekatan *transfer learning* untuk tugas klasifikasi gambar, khususnya membedakan antara gambar kucing dan anjing. Proyek ini menggunakan dataset "Cats vs Dogs" dan mengimplementasikan pipeline klasifikasi gambar yang melibatkan tahap preprocessing, augmentasi data, membangun model transfer learning dengan MobileNetV2, serta melatih model dengan strategi dua fase (frozen dan fine-tuning).

## Dataset yang Digunakan

**Dataset:** Cats vs Dogs Dataset dari Microsoft
**Link:** https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip

**Penjelasan Dataset:**
Dataset ini merupakan versi yang disederhanakan dari dataset Cats vs Dogs yang populer di Kaggle. Dataset ini berisi ribuan gambar yang dibagi menjadi dua kategori utama: gambar kucing dan gambar anjing. Dataset ini umum digunakan sebagai benchmark awal untuk tugas klasifikasi gambar binary.

- **Kelas:** Dataset ini memiliki 2 kelas, yaitu Kucing (direpresentasikan dengan label 0) dan Anjing (direpresentasikan dengan label 1). Ini menjadikannya masalah klasifikasi biner.
- **Jumlah Data yang Digunakan:** Dalam proyek ini, kami mencoba memuat hingga **1500 gambar kucing** dan **1500 gambar anjing**, sehingga total data yang dimuat **maksimal 3000 gambar** untuk pelatihan dan pengujian model. Pembatasan jumlah ini dilakukan untuk efisiensi komputasi dan sumber daya.

## Preprocessing dan Data Loading

Tahap preprocessing dan data loading dalam proyek ini dilakukan dengan pendekatan sederhana namun efektif menggunakan library OpenCV (`cv2`) dan NumPy.

- **Loading Data:** Gambar dimuat dari folder 'Cat' dan 'Dog'. File gambar dengan ekstensi umum (.png, .jpg, .jpeg) diproses.
- **Filter Ukuran Minimum:** Gambar dengan dimensi (tinggi atau lebar) kurang dari 50 piksel dilewati untuk menghindari gambar yang terlalu kecil atau rusak.
- **Resize gambar ke 150x150 piksel:**
    - **Alasan Penggunaan:** Model jaringan saraf tiruan, terutama model yang dilatih sebelumnya seperti MobileNetV2 yang digunakan dalam transfer learning, memerlukan input gambar dengan ukuran yang konsisten. Me-resize semua gambar ke 150x150 piksel memastikan bahwa semua input ke model memiliki dimensi yang sama, yang merupakan persyaratan dasar untuk pelatihan batch.
- **Konversi ke RGB:** Gambar dikonversi dari format BGR (default OpenCV) ke RGB, yang merupakan format umum yang diharapkan oleh banyak model deep learning, termasuk yang dilatih dengan library seperti TensorFlow/Keras.
- **Normalisasi Piksel:** Nilai piksel gambar dinormalisasi dengan membaginya dengan 255.0.
    - **Alasan Penggunaan:** Normalisasi ini mengubah rentang nilai piksel dari [0, 255] menjadi [0.0, 1.0]. Ini penting karena nilai piksel yang dinormalisasi membantu optimasi model bekerja lebih baik dan cepat. Skalasi input ke rentang kecil dan konsisten mencegah beberapa fitur mendominasi yang lain dan membantu konvergensi proses pelatihan.
- **Pengumpulan Data dan Label:** Gambar yang diproses dan labelnya (0 untuk Kucing, 1 untuk Anjing) dikumpulkan ke dalam list dan kemudian diubah menjadi array NumPy.

Setelah data dimuat, data diacak menggunakan `np.random.permutation` untuk memastikan distribusi kelas yang merata sebelum pembagian data.

## Pembagian Data

Dataset yang telah dimuat kemudian dibagi menjadi tiga subset untuk keperluan pelatihan, validasi, dan pengujian model:

- **Data Pelatihan (Training Set):** Digunakan untuk melatih model.
- **Data Validasi (Validation Set):** Digunakan selama pelatihan untuk mengevaluasi kinerja model pada data yang belum pernah dilihat sebelumnya dan membantu dalam penyetelan hyperparameter serta mendeteksi overfitting.
- **Data Pengujian (Test Set):** Digunakan *setelah* model selesai dilatih untuk memberikan evaluasi akhir yang tidak bias terhadap kinerja model pada data yang sepenuhnya baru.

Pembagian data dilakukan menggunakan `train_test_split` dari `sklearn.model_selection` dengan perbandingan:

- 70% data untuk pelatihan (`X_train`, `y_train`)
- 30% data sementara (`X_temp`, `y_temp`)

Kemudian, data sementara (`X_temp`, `y_temp`) dibagi lagi menjadi:

- 50% data untuk validasi (`X_val`, `y_val`) dari `X_temp` (yang setara dengan 15% dari total data)
- 50% data untuk pengujian (`X_test`, `y_test`) dari `X_temp` (yang setara dengan 15% dari total data)

Parameter `random_state=42` digunakan untuk memastikan bahwa pembagian data bersifat reproducible (akan menghasilkan pembagian yang sama setiap kali kode dijalankan). Parameter `stratify=y` digunakan untuk memastikan bahwa proporsi kelas (kucing vs anjing) dipertahankan dalam setiap subset data, yang penting untuk dataset biner seperti ini.

## Augmentasi Data

Untuk meningkatkan generalisasi model dan mengurangi overfitting, teknik augmentasi data diterapkan pada data pelatihan. Augmentasi data secara artifisial memperbesar dataset pelatihan dengan membuat versi modifikasi dari gambar yang ada.

Augmentasi data dilakukan menggunakan `ImageDataGenerator` dari `tf.keras.preprocessing.image` dengan konfigurasi berikut:

- `rotation_range=20`: Merotasi gambar secara acak hingga 20 derajat.
- `width_shift_range=0.2`: Menggeser gambar secara horizontal hingga 20% dari total lebar.
- `height_shift_range=0.2`: Menggeser gambar secara vertikal hingga 20% dari total tinggi.
- `horizontal_flip=True`: Membalik gambar secara horizontal secara acak.
- `zoom_range=0.2`: Memperbesar atau memperkecil gambar secara acak hingga 20%.
- `brightness_range=[0.8, 1.2]`: Mengubah kecerahan gambar secara acak antara 80% dan 120% dari kecerahan asli.
- `fill_mode='nearest'`: Mengisi piksel yang hilang setelah transformasi (seperti rotasi atau pergeseran) menggunakan nilai piksel terdekat.

Augmentasi ini hanya diterapkan pada data pelatihan untuk menghindari "kebocoran" informasi dari data pelatihan ke data validasi/pengujian.

## Arsitektur Model

Proyek ini menggunakan pendekatan *Transfer Learning* dengan memanfaatkan arsitektur **MobileNetV2** yang telah dilatih sebelumnya pada dataset ImageNet. Transfer learning memungkinkan model untuk menggunakan fitur-fitur umum yang telah dipelajari dari dataset besar (ImageNet) dan menerapkannya pada tugas yang baru (klasifikasi kucing vs anjing), yang sangat efektif ketika dataset baru berukuran relatif kecil.

Arsitektur model terdiri dari:

1.  **Base Model (MobileNetV2):** Arsitektur MobileNetV2 dimuat tanpa lapisan klasifikasi teratas (`include_top=False`) dan menggunakan bobot yang telah dilatih di ImageNet (`weights='imagenet'`). Lapisan input disesuaikan dengan ukuran gambar yang telah di-resize (150x150x3).
2.  **GlobalAveragePooling2D:** Lapisan ini mengurangi dimensi spasial output dari base model menjadi vektor tunggal dengan mengambil rata-rata spasial di setiap channel. Ini membantu mengurangi jumlah parameter dan bertindak sebagai jembatan antara fitur konvolusional dan lapisan Dense.
3.  **BatchNormalization:** Normalisasi batch diterapkan untuk menstabilkan proses pelatihan dan memungkinkan penggunaan *learning rate* yang lebih tinggi.
4.  **Dropout (0.3):** Lapisan Dropout dengan tingkat 0.3 secara acak menonaktifkan 30% neuron selama pelatihan untuk mencegah overfitting.
5.  **Dense (128, activation='relu'):** Lapisan Dense dengan 128 unit dan fungsi aktivasi ReLU untuk belajar kombinasi fitur tingkat tinggi.
6.  **BatchNormalization:** Normalisasi batch lagi setelah lapisan Dense pertama.
7.  **Dropout (0.2):** Lapisan Dropout dengan tingkat 0.2.
8.  **Dense (1, activation='sigmoid'):** Lapisan output Dense tunggal dengan fungsi aktivasi Sigmoid. Output dari lapisan ini adalah probabilitas bahwa gambar termasuk dalam kelas positif (Anjing, label 1).

## Strategi Pelatihan

Model dilatih menggunakan strategi dua fase untuk memanfaatkan transfer learning secara efektif:

### Fase 1: Melatih Lapisan Atas (Frozen Base Model)

- Pada fase ini, **lapisan dari base model (MobileNetV2) dibekukan** (`base_model.trainable = False`). Ini berarti bobot dari MobileNetV2 tidak akan diperbarui selama pelatihan.
- Hanya **lapisan yang baru ditambahkan di atas base model** (GlobalAveragePooling2D, BatchNormalization, Dense, Dropout) yang dilatih.
- Model dikompilasi dengan **optimizer Adam** dan **learning rate awal 0.001**.
- **Loss function** yang digunakan adalah `binary_crossentropy`, yang cocok untuk tugas klasifikasi biner.
- **Metric** yang dipantau adalah `accuracy`.
- Pelatihan dilakukan selama **15 epoch**.
- **Callbacks** EarlyStopping dan ReduceLROnPlateau digunakan untuk memantau validasi akurasi/loss dan menyesuaikan learning rate.

Tujuan dari fase ini adalah untuk melatih lapisan-lapisan baru agar dapat menginterpretasikan fitur-fitur tingkat tinggi yang diekstraksi oleh MobileNetV2 yang telah dilatih sebelumnya.

### Fase 2: Fine-tuning Base Model

- Setelah Fase 1, **base model (MobileNetV2) dibuka kembali untuk pelatihan** (`base_model.trainable = True`).
- Namun, **hanya lapisan-lapisan terakhir dari base model yang di-fine-tune**. Lapisan-lapisan awal biasanya mempelajari fitur-fitur yang sangat umum (tepi, sudut) dan tidak perlu banyak diubah, sementara lapisan yang lebih dalam mempelajari fitur yang lebih spesifik. Fine-tuning dimulai dari lapisan ke-100 dari MobileNetV2.
- Model **dikompilasi ulang** dengan **learning rate yang jauh lebih kecil (0.0001)**. Learning rate yang lebih kecil penting saat fine-tuning untuk menghindari perusakan bobot yang telah dipelajari sebelumnya dan memungkinkan penyesuaian yang lebih halus.
- Loss function dan metrics tetap sama.
- Pelatihan dilanjutkan selama **10 epoch tambahan** (total epoch untuk kedua fase adalah jumlah epoch Fase 1 + jumlah epoch Fase 2).
- Callbacks EarlyStopping dan ReduceLROnPlateau tetap aktif.

Tujuan dari fase ini adalah untuk sedikit menyesuaikan bobot dari lapisan-lapisan akhir MobileNetV2 agar lebih spesifik pada tugas klasifikasi kucing vs anjing, sehingga berpotensi meningkatkan kinerja model lebih lanjut.

## Hasil dan Evaluasi

Metrik evaluasi yang dilaporkan meliputi:

- **Test Loss:** Nilai fungsi kerugian pada dataset pengujian. Menunjukkan seberapa baik model memprediksi probabilitas kelas.
- **Test Accuracy:** Proporsi prediksi yang benar pada dataset pengujian. Menunjukkan seberapa sering model memprediksi kelas yang benar.
- **Classification Report:** Ringkasan metrik kinerja per kelas, termasuk:
    - **Precision:** Rasio true positive terhadap total positif yang diprediksi (kemampuan model untuk tidak melabeli sampel negatif sebagai positif).
    - **Recall (Sensitivity):** Rasio true positive terhadap total positif aktual (kemampuan model untuk menemukan semua sampel positif).
    - **F1-Score:** Rata-rata harmonik dari precision dan recall.
    - **Support:** Jumlah sampel aktual di setiap kelas dalam dataset pengujian.
- **Confusion Matrix:** Tabel yang menunjukkan jumlah true positive, true negative, false positive, dan false negative. Ini membantu visualisasi kinerja model dalam membedakan antara kelas.

Target akurasi yang diharapkan untuk proyek ini adalah antara 85-92%.

PERFORMANCE METRICS:
Test Accuracy: 0.9511 (95.11%)
Test Loss: 0.2117
