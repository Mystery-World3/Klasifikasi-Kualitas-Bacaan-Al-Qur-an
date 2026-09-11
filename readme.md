# Klasifikasi Kualitas Bacaan Al-Qur'an (Tahsin AI)

> **Skripsi:** Representasi Fitur Audio Tahsin Al-Qur'an Menggunakan Semi-Supervised Contrastive Learning.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-red)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-green)
![Hugging%20Face](https://img.shields.io/badge/Model-Hugging%20Face-yellow)

Sistem berbasis **Deep Learning** untuk mengklasifikasikan kualitas bacaan Al-Qur'an (Tahsin) secara otomatis dari rekaman audio. Pendekatan yang digunakan adalah **Semi-Supervised Contrastive Learning** dengan representasi **Log Mel Spectrogram** dan backbone **ResNet-18**.

## Live Demo

Coba aplikasi secara langsung:

**[Buka Tahsin AI](https://klasifikasi-kualitas-bacaan-al-qur-an.streamlit.app/)**

Model inference pada aplikasi diambil secara otomatis dari Hugging Face Hub sehingga bobot model tidak perlu disimpan di repository GitHub.

**Model:** [Mysteryworld3/quran-recitation-classifier](https://huggingface.co/Mysteryworld3/quran-recitation-classifier)

## Gambaran Sistem

```text
Audio WAV
   ↓
Resampling 22050 Hz
   ↓
Pembagian audio menjadi chunk 3 detik
   ↓
Log Mel Spectrogram
   ↓
Normalisasi + Resize 64 × 64
   ↓
ResNet-18
   ↓
Classification Head
   ↓
Probabilitas 5 kelas
   ↓
Agregasi probabilitas seluruh chunk
   ↓
Prediksi kualitas bacaan
```

## Kategori Kualitas Bacaan

Model mengklasifikasikan audio ke dalam 5 kategori:

| Kelas | Keterangan |
|---|---|
| **Mumtaz** | Kualitas bacaan sangat baik |
| **Jayyid Jiddan** | Kualitas bacaan sangat baik dengan kesalahan relatif minim |
| **Jayyid** | Kualitas bacaan baik namun masih memerlukan perhatian pada detail tertentu |
| **Maqbul** | Kualitas bacaan cukup dan masih memerlukan latihan |
| **Rasib** | Kualitas bacaan memerlukan peningkatan dan latihan lebih lanjut |

## Metode

### 1. Preprocessing Audio

Audio diproses menggunakan konfigurasi utama berikut:

- Sample rate: **22.050 Hz**
- Durasi input model: **3 detik**
- Mel bins: **64**
- FFT: **2048**
- Hop length: **512**
- Representasi: **Log Mel Spectrogram**
- Ukuran input CNN: **64 × 64**

Pada inference, audio panjang dibagi menjadi beberapa potongan berdurasi sekitar 3 detik. Setiap potongan dianalisis secara terpisah kemudian hasil probabilitasnya dirata-ratakan untuk memperoleh prediksi akhir.

### 2. Semi-Supervised Contrastive Learning

Training terdiri dari dua tahap:

**Stage 1 — Contrastive Pre-training**

Model mempelajari representasi fitur audio menggunakan data audio tanpa label melalui **contrastive learning** dan **NT-Xent loss**.

**Stage 2 — Fine-tuning**

Representasi yang telah dipelajari kemudian digunakan untuk klasifikasi lima kelas kualitas bacaan menggunakan classification head.

### 3. Arsitektur Model

```text
Log Mel Spectrogram (1 × 64 × 64)
               ↓
          ResNet-18
               ↓
        Feature Embedding
          ↙           ↘
 Projection Head   Classification Head
   (Stage 1)          (Stage 2)
```

Model menggunakan input satu channel karena representasi audio berbentuk spectrogram grayscale. Struktur backbone, projection head, dan classification head dipertahankan konsisten antara training dan inference.

## Hasil Evaluasi

Eksperimen dilakukan pada lima seed untuk melihat konsistensi performa model.

| Seed | Accuracy | Precision | Recall | F1-Score |
|---:|---:|---:|---:|---:|
| 42 | 80.61% | 80.69% | 80.61% | 80.60% |
| 52 | **82.50%** | **82.52%** | **82.50%** | **82.48%** |
| 62 | 80.91% | 81.07% | 80.91% | 80.94% |
| 72 | 80.85% | 80.81% | 80.85% | 80.82% |
| 82 | 80.55% | 80.74% | 80.55% | 80.56% |
| **Rata-rata** | **81.08%** | **81.17%** | **81.08%** | **81.08%** |

> Catatan: Seed final untuk deployment menggunakan **Seed 82**, dipilih berdasarkan performa validation sehingga pemilihan model tidak didasarkan pada hasil test set.

## Model

Checkpoint model untuk inference disimpan di Hugging Face Hub:

**Repository:** `Mysteryworld3/quran-recitation-classifier`

**Checkpoint deployment:** `classifier_seed_82.pth`

Repository model:

https://huggingface.co/Mysteryworld3/quran-recitation-classifier

Aplikasi Streamlit mengunduh checkpoint tersebut secara otomatis menggunakan `huggingface_hub`.

## Dataset

Dataset penelitian berasal dari rekaman bacaan Surah Maryam ayat 1–10.

Dataset yang digunakan untuk eksperimen tersedia melalui Kaggle:

https://www.kaggle.com/datasets/raffaarvel/dataset-maryam-1-10-potong

## Struktur Repository

```text
Klasifikasi-Kualitas-Bacaan-Al-Qur-an/
│
├── src/
│   ├── config.py
│   ├── dataset.py
│   ├── loss.py
│   ├── model.py
│   └── utils.py
│
├── app_demo.py
├── predict.py
├── evaluate_model.py
├── train.py
├── train_stage1_contrasive.py
├── prepare_dataset.py
├── split_dataset.py
├── plot_embeddings.py
├── plot_history.py
├── requirements.txt
├── .gitignore
└── readme.md
```

> Bobot model `.pth`, dataset mentah, dan output eksperimen tidak disimpan di repository GitHub dan dikelola terpisah sesuai kebutuhan training/deployment.

## Instalasi

### 1. Clone Repository

```bash
git clone https://github.com/Mystery-World3/Klasifikasi-Kualitas-Bacaan-Al-Qur-an.git
cd Klasifikasi-Kualitas-Bacaan-Al-Qur-an
```

### 2. Buat Virtual Environment

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux / macOS:**

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Penggunaan

### 1. Web App

Jalankan aplikasi:

```bash
streamlit run app_demo.py
```

Setelah browser terbuka, upload file audio **WAV**, kemudian jalankan analisis.

Aplikasi akan:

1. memuat checkpoint dari Hugging Face;
2. membaca audio pada 22.050 Hz;
3. membagi audio menjadi potongan sekitar 3 detik;
4. membentuk Log Mel Spectrogram;
5. melakukan klasifikasi menggunakan ResNet-18;
6. mengagregasikan probabilitas seluruh potongan; dan
7. menampilkan kelas akhir, confidence, distribusi probabilitas, detail setiap potongan, serta visualisasi spectrogram.

### 2. Prediksi melalui Terminal

Gunakan `predict.py` untuk menguji satu file audio dari command line:

```bash
python predict.py --audio "path/to/audio.wav"
```

Contoh Windows:

```powershell
python predict.py --audio "data/test_audio.wav"
```

Script akan mengambil `classifier_seed_82.pth` dari Hugging Face secara otomatis dan menampilkan hasil prediksi per chunk serta kesimpulan akhir.

### 3. Evaluasi Model

Untuk mengevaluasi model pada test split:

```bash
python evaluate_model.py
```

### 4. Training

Stage 1:

```bash
python train_stage1_contrasive.py
```

Stage 2:

```bash
python train.py
```

## Dependensi Utama

- **PyTorch** — framework deep learning
- **Torchvision** — ResNet-18
- **Librosa** — pemrosesan audio dan Mel Spectrogram
- **SoundFile** — pembacaan/penulisan audio
- **Scikit-learn** — evaluasi model
- **Pandas / NumPy** — pengolahan data
- **Matplotlib / Seaborn** — visualisasi
- **Streamlit** — antarmuka aplikasi
- **Hugging Face Hub** — distribusi checkpoint model

## Catatan Reproduksibilitas

Untuk menjaga konsistensi inference dengan model yang dilatih:

- preprocessing audio mengikuti konfigurasi project;
- input spectrogram menggunakan satu channel;
- inference tidak menggunakan augmentasi acak seperti time shift;
- checkpoint deployment menggunakan `classifier_seed_82.pth`;
- model dibangun dengan `pretrained=False` saat inference agar tidak melakukan download bobot ImageNet tambahan.

## Penulis

**Muhammad Mishbahul Muflihin**  
Program Studi Teknik Informatika  
Universitas Darussalam Gontor


