# Flower Classifier Backend (Flask + TensorFlow)

API REST para classificação de flores utilizando um modelo treinado em TensorFlow.
Este backend recebe uma imagem enviada via multipart/form-data e retorna:

- A classe prevista (daisy, dandelion, roses, sunflowers, tulips)
- A confiança da predição
- Probabilidades para cada classe

---

## Tecnologias

- Python 3.10+
- Flask
- TensorFlow 2.x (SavedModel)
- NumPy
- Pillow (PIL)

---

##  Estrutura do projeto

flower_classifier_backend/
│
├── app.py                # Servidor Flask (API)

├── train_model.py        # Script de treinamento do modelo

├── flower_model/         # Modelo salvo (TensorFlow SavedModel)

│   ├── assets/

│   ├── variables/

│   ├── saved_model.pb

│   └── fingerprint.pb

├── class_names.json      # Lista das classes treinadas

├── requirements.txt      # Dependências do projeto

└── README.md             # Documentação do backend

```yaml
## Como rodar o backend
```
### 1. Criar venv
python -m venv venv

### 2. Ativar
venv\Scripts\activate

```shell
### 3. Instalar dependências
```
pip install -r requirements.txt

```shell
### 4. Rodar a API
```
python app.py

```css
A API ficará disponível em:
```
http://127.0.0.1:5000/predict

```yaml
---

## Testando a API

Use o Postman:

- Método: **POST**
- Body: **form-data**
- Key: `image`  
- Tipo: **File**
- Value: (selecione uma imagem)

Exemplo de resposta:

json
{
  "class": "dandelion",
  "confidence": 0.9987,
  "probabilities": {
    "daisy": 0.0011,
    "dandelion": 0.9987,
    "roses": 0.0001,
    "sunflowers": 0.00001,
    "tulips": 0.00008
  }
}
```
