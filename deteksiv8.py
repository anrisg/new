import cv2  # type: ignore
import time
import torch
import torchvision
from ultralytics import YOLO  # type: ignore

# Cek apakah torch dan torchvision telah terinstall dengan benar
def check_torch_installation():
    print(f"Torch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA tidak tersedia, menggunakan CPU.")

check_torch_installation()

# Load model YOLOv8 hasil training
model_path = r"D:\yov8\last.pt" # Ubah ke 'last.pt' jika diperlukan
model = YOLO(model_path)

# Inisialisasi kamera laptop
cap = cv2.VideoCapture(0)

# Cek apakah kamera berhasil terbuka
if not cap.isOpened():
    print("Tidak dapat membuka kamera laptop.")
    exit()

# Variabel untuk menghitung FPS
prev_time = 0

# Loop untuk membaca frame dari kamera
while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca frame dari kamera.")
        break

    # Dapatkan waktu saat ini untuk menghitung FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Deteksi objek menggunakan model hasil training dengan threshold confidence 0.7
    results = model.predict(frame, conf=0.7)

    # Salin frame untuk anotasi
    annotated_frame = frame.copy()

    # Loop melalui setiap deteksi dan tambahkan bounding box serta nama kelas
    for result in results:
        if hasattr(result, 'boxes') and result.boxes is not None:
            for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
                x1, y1, x2, y2 = map(int, box)  # Koordinat bounding box
                class_id = int(cls)  # ID kelas

                # Ambil nama kelas berdasarkan model hasil training
                class_name = model.names.get(class_id, "Unknown")

                label = f"{class_name} ({x1}, {y1})"

                # Gambar bounding box dan label pada frame
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

    # Tambahkan FPS di tampilan
    cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Tampilkan hasil deteksi
    cv2.imshow('Deteksi Sampah dengan Model Custom', annotated_frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Lepaskan kamera dan tutup semua jendela
cap.release()
cv2.destroyAllWindows()
