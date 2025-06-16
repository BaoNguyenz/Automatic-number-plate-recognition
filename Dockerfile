# Sử dụng một base image Python chính thức, phiên bản 3.10-slim
FROM python:3.10-slim

# Cài đặt các thư viện hệ thống mà OpenCV cần
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Cài đặt Poetry
RUN pip install poetry

# Thiết lập thư mục làm việc bên trong container
WORKDIR /app

# Cấu hình Poetry để không tạo virtualenv bên trong container
RUN poetry config virtualenvs.create false

# Copy file cấu hình và lock file trước để tận dụng cache của Docker
COPY pyproject.toml poetry.lock ./

# Cài đặt các dependencies từ lock file (chỉ cài production dependencies)
RUN poetry install --no-root --no-dev

# Copy toàn bộ mã nguồn của dự án vào thư mục làm việc
COPY . .

# Mở cổng 8501 để Streamlit có thể truy cập từ bên ngoài
EXPOSE 8501

# Lệnh để chạy ứng dụng khi container khởi động
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]