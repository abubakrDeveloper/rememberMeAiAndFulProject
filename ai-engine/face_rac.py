import cv2
import mediapipe as mp
import threading
import requests
import base64
import time

# Serverning API manzili (Testing uchun localhost ko'rsatilgan)
API_URL = "http://localhost:5000/api/face"

# Mediapipe Face Detection kutubxonasini yuklash
mp_face_detection = mp.solutions.face_detection

def send_to_server(image_base64_data):
    """
    Ma'lumotlarni backend serverga integratsiya qilish.
    """
    try:
        # JSON so'rov shakli
        payload = {
            "image": image_base64_data,
            "timestamp": time.time()
        }
        
        # Serverga jo'natish
        response = requests.post(API_URL, json=payload, timeout=5)
        
        if response.status_code == 200:
            print(f"[YUBORILDI] Yuz serverga yetkazildi. Javob: {response.status_code}")
        else:
            print(f"[XATOLIK] Server xatosi: {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        print(f"[XATOLIK] Serverga ulanib bo'lmadi: {e}")

def process_face(frame, face_detection):
    """
    Yuzni tahlil qilish va tanish. Yuz qirqiladi va jo'natiladi.
    """
    # Mediapipe ishlashi uchun rasmni RGB formatga o'tkazish
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)

    h_img, w_img, _ = frame.shape

    if results.detections:
        for detection in results.detections:
            # Bounding box koordinatalari (0 dan 1 gacha nisbiy qiymatlarda)
            bboxC = detection.location_data.relative_bounding_box
            
            x = int(bboxC.xmin * w_img)
            y = int(bboxC.ymin * h_img)
            w = int(bboxC.width * w_img)
            h = int(bboxC.height * h_img)

            # Ekranga va kadrdan chiqib ketish xatolarini oldini olish
            x, y = max(0, x), max(0, y)
            w = min(w_img - x, w)
            h = min(h_img - y, h)

            # Yuz qismini to'rtburchak orqali belgilab chizish
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Yuz qismini qirqib olish (crop)
            face_crop = frame[y:y+h, x:x+w]

            if face_crop.size != 0:
                # Kadrni JPEG formatiga siqib o'tkazish (sifat 90%)
                success, buffer = cv2.imencode('.jpg', face_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                
                if success:
                    # JSON shaklida serverga yuborish uchun Base64 formatga aylantirish
                    base64_image = base64.b64encode(buffer).decode('utf-8')

                    # Tarmoq kechikishlarini oldini olish uchun alohida threading orqali POST qilish
                    thread = threading.Thread(target=send_to_server, args=(base64_image,))
                    thread.start()

    return frame

def capture_frame():
    """
    Kameradan ruxsat olish va tasvirni o'qish.
    """
    # Tizimdagi birlamchi kamerani (ID: 0) ishga tushirish
    cap = cv2.VideoCapture(0)

    # Xavfsizlik: Kameradan ruxsat yo'qligi yoki bandligi tekshiruvi
    if not cap.isOpened():
        print("[XATOLIK] Kameraga ulanib bo'lmadi. Kamera band bo'lishi yoki ruxsat yo'q bo'lishi mumkin.")
        return

    print("[MA'LUMOT] Kamera ishga tushdi. Yuzingizni kameraga qarating (Chiqish uchun 'q' tugmasini bosing).")

    # Mediapipe Face Detection sozlamalari (min ishqonch darajasi 0.7 qilib belgilandi)
    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.7) as face_detection:
        while True:
            # Kadrni o'qish
            success, frame = cap.read()
            
            if not success:
                print("[XATOLIK] Kadrdan o'qishda uzilish bo'ldi.")
                break

            # Kadrni yuz uchun tahlil qilish
            processed_frame = process_face(frame, face_detection)

            # Kadrni foydalanuvchiga real vaqtda ko'rsatish
            cv2.imshow("Kamera - Yuzni Aniqlash (AI Engine)", processed_frame)

            # 1 ms kutish va 'q' tugmasi orqali jarayonni bekor qilish
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Dastur yakunida kamerani bo'shatish va oynalarni yopish
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_frame()
