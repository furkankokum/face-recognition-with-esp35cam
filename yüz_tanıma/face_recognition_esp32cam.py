import cv2
import math
from sklearn import neighbors
import os
import os.path
import pickle
from PIL import Image, ImageDraw
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import numpy as np
import xlsxwriter
from datetime import datetime, date

names = []
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "JPG"}

def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
    """
    KNN algoritması için eğitim fonksiyonu
    train_dir: Tanınan insanları içeren klasörün üst klasörü
    model_save_path: modelin diskte kaydedilme yolu
    n_neighbors: Sınıflandırmada komşuların sayısı
    knn_algo: knn.default'u desteklemek için temel alınan veri yapısı ball_tree
    verbose: eğitimin ayrıntısı
    return: Verilen dataya göre eğitilmiş knn sınıflandırıcısı
    
    """
    X = []
    y = []

    # Eğitim verisindeki her kişi için for döngüsü
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue

        # Bir kişinin tüm fotoğrafları için for döngüsü
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image)

            if len(face_bounding_boxes) != 1:
                # Eğer resimde birden fazla yüz varsa resimi atla
                if verbose:
                    print("Görsel {} eğitim için uygun değil: {}".format(img_path, "Yüz Bulunamadı" if len(face_bounding_boxes) < 1 else "Görselde birden fazla yüz bulundu"))
            else:
                # Eğitim setine mevcut görüntü için yüz kodlaması ekle
                X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                y.append(class_dir)

    # KNN sınıflandırıcısında ağırlıklandırma için kaç tane komşu kullanılacağını belirleyin
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            print(f"Otomatik olarak {n_neighbors} komşu seçildi:")

    # KNN sınıflandırıcısı oluştur ve eğit
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)

    # Eğitilmiş sınıflandırıcıyı kaydet
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)

    return knn_clf


def predict(X_frame, knn_clf=None, model_path=None, distance_threshold=0.5):
    """
    Eğitilmiş KNN sınıflandırıcısını kullanarak verilen görseldeki yüzleri tanır.

    X_frame: Tanımanın yapılacağı frame
    knn_clf: knn sınıflandırıcısı
    model_path: knn sınıflandırıcısının diskte bulunduğu konum
    distance_threshold: yüz sınıflandırması için mesafe eşiği. ne kadar büyükse,
    bilinmeyen bir kişiyi bilinen kişi olarak yanlış sınıflandırma şansı o kadar fazladır
    return: resimde tanınan yüzler için adların ve yüz konumlarının bir listesi:
    Tanınmayan kişilerin yüzleri için 'bilinmeyen' adı döndürülür.

    """
    if knn_clf is None and model_path is None:
        raise Exception("KNN sınıflandırıcısı knn_clf veya model_save_path ile verilmelidir.")

    # Eğer verilmiş ise eğitilmiş KNN modelini yükle.
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    X_face_locations = face_recognition.face_locations(X_frame)

    # Eğer görselde yüz bulunamazsa boş döndür.
    if len(X_face_locations) == 0:
        return []

    # Test görüntüsündeki yüzler için kodlamaları bul.
    faces_encodings = face_recognition.face_encodings(X_frame, known_face_locations=X_face_locations)

    # En iyi eşleşmeyi bulmak için KNN modelini kullan.
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

    # Sınıfları tahmin et ve eşik dahilinde olmayan sınıflandırmaları kaldır
    return [(pred, loc) if rec else ("Unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]


def log_to_file(name):
    """
    Tanınan yüzün kime ait olduğunu tarih ve saat bilgileri ile bir excel dosyasına yazdırır.
    name: Tanınan yüzün ait olduğu kişi
    """
    #Eğer tahmin edilen isim isimler listesinde değilse ekle ve excel dosyasına yazdır.
    if name not in names:
        names.append(name)
        worksheet.write(f'A{names.index(name) + 1}', name)
        worksheet.write(f'B{names.index(name) + 1}', date.today().strftime("%d/%m/%Y"))
        worksheet.write(f'C{names.index(name) + 1}', datetime.now().strftime('%H:%M:%S'))
    #Eğer listede ise kaydedilmiş tarih ve saati güncelle.
    else:
        worksheet.write(f'B{names.index(name) + 1}', date.today().strftime("%d/%m/%Y"))
        worksheet.write(f'C{names.index(name) + 1}', datetime.now().strftime('%H:%M:%S'))


def show_prediction_labels_on_image(frame, predictions):
    """
    Yüz tanıma sonuçlarını görsel olarak gösterir.

    frame: Tanıma sonuçlarının gösterildiği frame
    predictions: predict fonksiyonunun sonucu
    return opencv resmi cv2.imshow fonksiyonuyla uyumlu olacak şekilde
    """
    pil_image = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil_image)

    for name, (top, right, bottom, left) in predictions:
        # Tahminleri tam ekran boyutuna getir.
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2
        # Pillow modülünü kullanarak yüzün etrafına kare çiz.
        draw.rectangle(((left, top), (right, bottom)), outline=(205, 51, 51))

        # Tanınan yüzün ait olduğu kişinin ismini dosyaya kaydet.
        log_to_file(name)
        name = name.encode("UTF-8")

        # Yüzün altına yazı için kutucuk çiz.
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(205, 51, 51), outline=(205, 51, 51))
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))

    # Çizim kütüphanesini hafızadan kaldır.
    del draw

    #Görseli open-cv formatında kaydet.
    opencvimage = np.array(pil_image)
    return opencvimage

if __name__ == "__main__":
    workbook = xlsxwriter.Workbook('log.xlsx')
    worksheet = workbook.add_worksheet()
    print("KNN sınıflandırıcısı eğitiliyor...")
    classifier = train("knn_examples/train", model_save_path="trained_knn_model.clf", n_neighbors=2)
    print("Eğitim tamamlandı!")
    # Hız için 30 frame arasından birisini işle.
    process_this_frame = 29
    print('Kamera kuruluyor!')
    url = "http://192.168.43.21:81/stream"
    cap = cv2.VideoCapture(url)
    while True:
        ret, frame = cap.read()
        if ret:
            img = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            process_this_frame = process_this_frame + 1
            if process_this_frame % 30 == 0:
                predictions = predict(img, model_path="trained_knn_model.clf")
            frame = show_prediction_labels_on_image(frame, predictions)
            cv2.imshow("camera", frame)
            if ord('q') == cv2.waitKey(10):
                cap.release()
                cv2.destroyAllWindows()
                workbook.close()
                exit(0)
