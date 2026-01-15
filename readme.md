# 🎙️ Klasifikasi Kualitas Bacaan Al-Qur'an (Tahsin AI)

> **Skripsi:** Representasi Fitur Audio Tahsin Qiroah Menggunakan Semi-Supervised Contrastive Learning.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-red)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-green)

Proyek ini adalah sistem berbasis **Deep Learning** yang dikembangkan untuk mengklasifikasikan kualitas bacaan Al-Qur'an (Tahsin) secara otomatis. Sistem ini mampu menganalisis file audio dan memberikan penilaian berdasarkan standar makhorijul huruf dan tajwid.

## 🌐 Live Demo
Cobalah aplikasi ini secara langsung tanpa perlu instalasi:
👉 **[Klik di sini untuk mencoba Aplikasi](https://klasifikasi-kualitas-bacaan-al-quran.streamlit.app/)**

## 📋 Fitur Utama

Sistem ini mengklasifikasikan audio ke dalam 5 kategori kualitas:
1.  🌟 **Mumtaz** 
2.  ✅ **Jayyid Jiddan** 
3.  👍 **Jayyid** 
4.  🆗 **Maqbul** 
5.  ⚠️ **Rosib** 

**Fitur Aplikasi:**
* **Web Interface (GUI):** Antarmuka berbasis Streamlit yang ramah pengguna untuk menilai rekaman secara langsung.
* **Batch Prediction:** Fitur untuk memproses ratusan file audio sekaligus dan mengekspor hasil analisisnya ke Excel/CSV.
* **Audio Preprocessing:** Otomatis melakukan *resampling*, *rechanneling*, dan *padding/truncating* sinyal audio.

## 📂 Struktur Direktori

Pastikan struktur folder Anda terlihat seperti ini:

```text
├── data/
│   └── unlabeled/      
├── models/
│   └── final_model_skripsi.pth  
├── src/
│   ├── config.py        
│   ├── model.py         
│   └── utils.py        
├── app_demo.py          
├── batch_predict.py    
├── predict.py          
├── requirements.txt     
└── README.md           

```

## 🚀 Cara Instalasi

1. **Clone Repository ini:**
```bash
git clone https://github.com/Mystery-World3/Klasifikasi-Kualitas-Bacaan-Al-Qur-an.git
cd Klasifikasi-Kualitas-Bacaan-Al-Qur-an

```


2. **Siapkan Environment (Disarankan):**
```bash
python -m venv venv

# Windows:
venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate

```


3. **Install Library yang Dibutuhkan:**
```bash
pip install -r requirements.txt

```


4. **Siapkan Model:**
Pastikan file model `final_model_skripsi.pth` sudah ada di dalam folder `models/`.

## 💻 Cara Penggunaan

### 1. Menjalankan Web App (Demo)

Gunakan mode ini untuk presentasi atau penggunaan interaktif yang mudah.

```bash
streamlit run app_demo.py

```

*Akan otomatis membuka browser. Upload file .wav, lalu klik tombol prediksi.*

### 2. Menjalankan Batch Prediction (Data Banyak)

Gunakan mode ini untuk merekap nilai dari banyak file audio sekaligus.

1. Masukkan semua file `.wav` ke folder `data/unlabeled/`.
2. Jalankan perintah:
```bash
python batch_predict.py

```


3. Hasil akan disimpan dalam file `laporan_hasil_prediksi.csv`.

### 3. Prediksi via Terminal (CLI)

Untuk pengujian cepat satu file.

```bash
python predict.py

```

*(Pastikan path file audio sudah disesuaikan di dalam script predict.py)*

## 🛠️ Teknologi yang Digunakan

* **Bahasa Pemrograman:** Python
* **Deep Learning Framework:** PyTorch
* **Audio Analysis:** Librosa, Torchaudio
* **Interface:** Streamlit
* **Data Manipulation:** Pandas, Numpy

## 👨‍💻 Penulis

**Muhammad Mishbahul Muflihin**

* Program Studi Teknik Informatika
* Universitas Darussalam Gontor
