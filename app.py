import streamlit as st
import yaml
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer, WebRtcMode
from pipeline import MainProcessor

# Cấu hình trang
st.set_page_config(page_title="Nhận dạng biển số xe", layout="wide")
st.title("🚀 ANPR - Nhận dạng biển số xe thời gian thực")
st.write("Hướng camera của bạn về phía phương tiện để bắt đầu nhận dạng.")

# Tải cấu hình và khởi tạo bộ xử lý
try:
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    st.error("Lỗi: Không tìm thấy file 'config.yaml'.")
    st.stop()

@st.cache_resource
def get_main_processor():
    """Tải và trả về instance của bộ xử lý chính."""
    return MainProcessor(config)

main_processor = get_main_processor()

# Lớp callback cho streamlit-webrtc để xử lý từng frame
class ANPRVideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        # Chuyển frame từ WebRTC thành ảnh OpenCV
        img = frame.to_ndarray(format="bgr24")
        
        # Gọi hàm xử lý chính từ pipeline.py
        processed_img = main_processor.process_frame(img)
        
        # Trả về frame đã xử lý để hiển thị trên trình duyệt
        return frame.from_ndarray(processed_img, format="bgr24")

# Khởi chạy WebRTC streamer để hiển thị luồng camera
webrtc_streamer(
    key="anpr-streamer",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=ANPRVideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

st.success("Khởi động camera thành công! Chờ một lát để tải model...")