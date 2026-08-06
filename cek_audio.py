import librosa

# Kita tes satu file dari folder mumtaz yang kemaren gagal
file_tester = "data/labeled/mumtaz/P001_05.wav"

print(f"🔍 Mencoba membaca file: {file_tester}")
try:
    y, sr = librosa.load(file_tester, sr=None)
    print("✅ BERHASIL! File-nya normal.")
except Exception as e:
    print(f"❌ GAGAL! Ini alasan dari Python:\n{e}")