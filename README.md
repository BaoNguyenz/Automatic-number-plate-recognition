# ANPR - Nhận dạng biển số xe thời gian thực

Dự án này là một ứng dụng web sử dụng Streamlit để nhận dạng biển số xe trong thời gian thực từ luồng camera. Hệ thống sử dụng YOLOv8 để phát hiện phương tiện và biển số, thuật toán SORT để theo dõi và EasyOCR để đọc ký tự. Toàn bộ ứng dụng được đóng gói bằng Docker để dễ dàng triển khai.

## ✨ Tính năng

- Nhận dạng biển số xe real-time từ webcam.
- Phát hiện phương tiện (xe hơi, xe máy...).
- Theo dõi đối tượng (tracking) để gán biển số cho xe tương ứng.
- Giao diện web tương tác được xây dựng bằng Streamlit.
- Đóng gói hoàn toàn bằng Docker và quản lý môi trường bằng Poetry.

## Chạy với Docker  

1.  **Build Docker image:**
    ```bash
    docker build -t anpr-app .
    ```

2.  **Chạy Docker container:**
    ```bash
    docker run --rm -p 8501:8501 anpr-app
    ```
    *Lưu ý:* Nếu bạn dùng Linux và muốn truy cập camera vật lý của máy, bạn có thể cần thêm cờ `--device=/dev/video0`. Tuy nhiên, với `streamlit-webrtc`, việc truy cập camera qua trình duyệt thường không cần bước này.

3.  Mở trình duyệt và truy cập `http://localhost:8501`.