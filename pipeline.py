import cv2
import numpy as np
import string
import easyocr
from ultralytics import YOLO
from sort.sort import Sort

class LicensePlateProcessor:
    """Chịu trách nhiệm cho mọi thứ liên quan đến OCR biển số xe."""
    def __init__(self, languages=['en'], use_gpu=False):
        self.reader = easyocr.Reader(languages, gpu=use_gpu)
        self.dict_char_to_int = {'O': '0', 'I': '1', 'J': '3', 'A': '4', 'G': '6', 'S': '5'}
        self.dict_int_to_char = {'0': 'O', '1': 'I', '3': 'J', '4': 'A', '6': 'G', '5': 'S'}

    def _license_complies_format(self, text):
        if len(text) != 7: return False
        if (text[0] in string.ascii_uppercase or text[0] in self.dict_int_to_char) and \
           (text[1] in string.ascii_uppercase or text[1] in self.dict_int_to_char) and \
           (text[2] in '0123456789' or text[2] in self.dict_char_to_int) and \
           (text[3] in '0123456789' or text[3] in self.dict_char_to_int) and \
           (text[4] in string.ascii_uppercase or text[4] in self.dict_int_to_char) and \
           (text[5] in string.ascii_uppercase or text[5] in self.dict_int_to_char) and \
           (text[6] in string.ascii_uppercase or text[6] in self.dict_int_to_char):
            return True
        return False

    def _format_license(self, text):
        license_plate_ = ''
        mapping = {0: self.dict_int_to_char, 1: self.dict_int_to_char, 2: self.dict_char_to_int, 3: self.dict_char_to_int,
                   4: self.dict_int_to_char, 5: self.dict_int_to_char, 6: self.dict_int_to_char}
        for j in range(7):
            if text[j] in mapping.get(j, {}):
                license_plate_ += mapping[j][text[j]]
            else:
                license_plate_ += text[j]
        return license_plate_

    def read_text(self, license_plate_crop):
        """Đọc ký tự từ ảnh crop của biển số."""
        detections = self.reader.readtext(license_plate_crop)
        for _, text, score in detections:
            text = text.upper().replace(' ', '')
            if self._license_complies_format(text):
                return self._format_license(text), score
        return None, None

class MainProcessor:
    """Điều phối toàn bộ quá trình xử lý cho mỗi khung hình."""
    def __init__(self, config):
        self.vehicle_detector = YOLO(config['weights']['vehicle_detector'])
        self.lp_detector = YOLO(config['weights']['license_plate_detector'])
        self.vehicle_tracker = Sort()
        self.lp_processor = LicensePlateProcessor(
            languages=config['ocr']['languages'], use_gpu=config['ocr']['use_gpu']
        )
        self.vehicle_classes = config['vehicle_classes']

    def _get_car(self, license_plate, vehicle_track_ids):
        x1, y1, x2, y2, _, _ = license_plate
        for xcar1, ycar1, xcar2, ycar2, car_id in vehicle_track_ids:
            if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
                return xcar1, ycar1, xcar2, ycar2, car_id
        return -1, -1, -1, -1, -1

    def process_frame(self, frame):
        """Xử lý một khung hình duy nhất và trả về frame đã được vẽ lên."""
        detections_ = []
        vehicle_detections = self.vehicle_detector(frame, verbose=False)[0]
        for det in vehicle_detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = det
            if int(class_id) in self.vehicle_classes:
                detections_.append([x1, y1, x2, y2, score])
        
        track_ids = self.vehicle_tracker.update(np.asarray(detections_))
        
        license_plates = self.lp_detector(frame, verbose=False)[0]
        for lp in license_plates.boxes.data.tolist():
            x1_lp, y1_lp, x2_lp, y2_lp, score_lp, _ = lp
            x_car, y_car, x2_car, y2_car, car_id = self._get_car(lp, track_ids)
            
            if car_id != -1:
                cv2.rectangle(frame, (int(x_car), int(y_car)), (int(x2_car), int(y2_car)), (0, 255, 0), 2)
                cv2.rectangle(frame, (int(x1_lp), int(y1_lp)), (int(x2_lp), int(y2_lp)), (255, 0, 0), 2)
                
                lp_crop = frame[int(y1_lp):int(y2_lp), int(x1_lp):int(x2_lp), :]
                lp_text, lp_score = self.lp_processor.read_text(lp_crop)
                
                if lp_text:
                    text_to_show = f"ID: {int(car_id)} - LP: {lp_text}"
                    cv2.putText(frame, text_to_show, (int(x_car), int(y_car) - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        return frame