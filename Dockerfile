# Usa uma imagem oficial do Python
FROM python:3.10-slim

# Evita problemas de buffering
ENV PYTHONUNBUFFERED=1

# Instala dependências do sistema (necessárias para TensorFlow CPU)
RUN apt-get update && apt-get install -y \
    build-essential \
    libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/*

# Cria diretório da aplicação
WORKDIR /app

# Copia requisitos primeiro (melhor para cache)
COPY requirements.txt .

# Instala dependências Python
RUN pip install --no-cache-dir -r requirements.txt

# Copia todo o projeto
COPY . .

# Expõe porta usada pelo Flask
EXPOSE 5000

# Comando para iniciar a API
CMD ["python", "app.py"]
