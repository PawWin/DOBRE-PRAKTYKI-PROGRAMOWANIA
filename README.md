# Automatic License Plate Recognition (ALPR)

System detekcji i OCR polskich tablic rejestracyjnych.

## Wymagania

- Python 3.10+
- CUDA (dla GPU acceleration)
- UV package manager
- Konto Kaggle z API key

## Instalacja

### 1. Zainstaluj UV (jeśli nie masz)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Stwórz środowisko i zainstaluj zależności

```bash
# Stwórz venv i zainstaluj podstawowe zależności
uv venv
source .venv/bin/activate

# Zainstaluj PaddlePaddle z GPU (CUDA 12.x)
uv pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

# Zainstaluj pozostałe zależności
uv pip install -e .
```

### 3. Skonfiguruj Kaggle API

```bash
# Stwórz folder .kaggle w home
mkdir -p ~/.kaggle

# Pobierz API key z https://www.kaggle.com/settings
# Stwórz plik ~/.kaggle/kaggle.json z zawartością:
# {"username":"YOUR_USERNAME","key":"YOUR_API_KEY"}

chmod 600 ~/.kaggle/kaggle.json
```

### 4. Pobierz dataset

```bash
python scripts/download_data.py
```

## Użycie

### Uruchom ewaluację

```bash
# Podstawowa ewaluacja (100 zdjęć)
python scripts/run_evaluation.py

# Więcej opcji
python scripts/run_evaluation.py --num-samples 200 --output results.json

# Bez GPU
python scripts/run_evaluation.py --no-gpu
```

### Parametry

- `--dataset-path PATH` - Ścieżka do datasetu (domyślnie: auto-detect)
- `--split {train,valid,test}` - Split datasetu (domyślnie: test)
- `--num-samples N` - Liczba próbek do ewaluacji (domyślnie: 100)
- `--output FILE` - Plik JSON na wyniki
- `--no-gpu` - Wyłącz GPU
- `--quiet` - Ciche działanie

## Struktura projektu

```
├── pyproject.toml          # Konfiguracja projektu i zależności
├── src/
│   ├── __init__.py
│   ├── detector.py         # Detekcja tablic (YOLOv8)
│   ├── ocr.py              # OCR (PaddleOCR)
│   ├── pipeline.py         # Główny pipeline
│   ├── evaluate.py         # Ewaluacja i metryki
│   └── utils.py            # Funkcje pomocnicze
└── scripts/
    ├── download_data.py    # Pobieranie datasetu
    └── run_evaluation.py   # Uruchomienie ewaluacji
```

## Metryki

- **Dokładność (accuracy)**: % prawidłowo odczytanych tablic
- **Czas przetwarzania**: czas dla 100 zdjęć w sekundach

### Funkcja oceny

Ocena końcowa wyliczana jest według wzoru:
- Dokładność ma wagę 0.7
- Czas ma wagę 0.3
- Minimalna dokładność: 60%
- Maksymalny czas: 60s dla 100 zdjęć

## Technologie

- **YOLOv8** - detekcja tablic rejestracyjnych
- **PaddleOCR** - rozpoznawanie tekstu
- **UV** - zarządzanie środowiskiem Python

## API + kolejka (FastAPI, RabbitMQ, SQLite)

W repo są gotowe obrazy dockerowe dla API oraz workera obsługującego kolejkę.

- Upewnij się, że model `LP-detection.pt` leży w katalogu głównym repo (jest kopiowany do obrazów).
- Dashboard RabbitMQ jest dostępny na porcie `15672` (login `guest` / `guest`).

### Uruchomienie (docker-compose)

```bash
docker compose up --build
```

Serwisy:
- `api` (FastAPI) na porcie `8000`
- `worker` (konsument kolejki)
- `rabbitmq` (kolejka + panel `http://localhost:15672`)

SQLite jest zapisywany w volume `./data/app.db`.

### Endpointy

- `POST /analyze` – synchroniczna analiza obrazu podanego jako URL:
  ```json
  { "image_url": "https://example.com/plate.jpg" }
  ```
- `POST /enqueue` – dodaje zadanie do kolejki, zwraca `job_id`:
  ```json
  { "image_url": "https://example.com/plate.jpg" }
  ```
- `GET /job/{job_id}` – status wyniku (`queued/running/done/error`) oraz rezultat (tekst tablicy, bbox, czasy).
- `GET /health` – prosty healthcheck.

### Ręczne testy (przykład)

```bash
# synchroniczna analiza
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"image_url":"https://example.com/plate.jpg"}'

# dodanie do kolejki
JOB_ID=$(curl -s -X POST http://localhost:8000/enqueue \
  -H "Content-Type: application/json" \
  -d '{"image_url":"https://example.com/plate.jpg"}' | jq -r .job_id)

# sprawdzenie statusu
curl http://localhost:8000/job/$JOB_ID
```

󰣇 ~/school/DOBRE-PRAKTYKI-PROGRAMOWANIA   automatic_plate_number_recognition  !? ❯ source .venv/bin/activate
python scripts/run_evaluation.py \
  --num-samples 195 \
  --use-yolo \
  --yolo-model /home/wajcha/school/DOBRE-PRAKTYKI-PROGRAMOWANIA/LP-detection.pt \
  --yolo-conf 0.25 \
  --yolo-imgsz 1056

https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fc8.alamy.com%2Fcomp%2F2R6F5T0%2Fwarsaw-poland-4-june-2023-polish-police-car-on-the-street-view-of-a-police-car-with-the-lettering-policja-police-patrol-car-parked-on-the-stree-2R6F5T0.jpg&f=1&nofb=1&ipt=d70b5d2eca44399dde30ec9fb038eb020d750daf95f08eafef2e0496b85b111b

uv run python -m api.worker 
uv run uvicorn api.main:app --host 0.0.0.0 --port 8000