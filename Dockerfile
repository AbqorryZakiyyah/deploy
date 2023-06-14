# Menggunakan base image Python
FROM python:3.8

# Mengatur environment variable
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Mengatur direktori kerja
WORKDIR /app

# Menyalin requirements.txt ke dalam container
COPY requirements.txt .

# Menginstal dependensi
RUN pip install -r requirements.txt

# Menyalin seluruh kode ke dalam container
COPY . .

# Menjalankan aplikasi Flask
CMD ["python", "app.py"]
