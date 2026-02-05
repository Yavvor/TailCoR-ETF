# Używamy lekkiego obrazu Pythona
FROM python:3.11-slim

# Ustawiamy katalog roboczy w kontenerze
WORKDIR /app

# Kopiujemy plik z zależnosciami
COPY requirements.txt .

# Instalujemy zależności
RUN pip install --no-cache-dir -r requirements.txt

# Kopiujemy resztę plików projektu (kod aplikacji, foldery ClassETF, Data itp.)
COPY . .

# Upewniamy się, że folder storage istnieje
RUN mkdir -p storage

# Zmienna środowiskowa dla portu
ENV PORT 5000

# Otwieramy port 5000
EXPOSE 5000

# Uruchamiamy aplikację używając serwera produkcyjnego Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]