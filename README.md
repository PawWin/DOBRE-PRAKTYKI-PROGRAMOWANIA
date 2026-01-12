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

󰣇 ~/school/DOBRE-PRAKTYKI-PROGRAMOWANIA   automatic_plate_number_recognition  !? ❯ source .venv/bin/activate
python scripts/run_evaluation.py \
  --num-samples 100 \
  --use-yolo \
  --yolo-model /home/wajcha/school/DOBRE-PRAKTYKI-PROGRAMOWANIA/LP-detection.pt \
  --yolo-conf 0.25 \
  --yolo-imgsz 1056