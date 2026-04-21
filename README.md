# Lab 2 - ML Pipeline & Experiment Tracking

Pipeline huấn luyện mô hình gợi ý rating phim sử dụng **Surprise**, theo dõi thực nghiệm bằng **MLflow**, và orchestration bằng **Airflow**.

## Mục tiêu

- Xây dựng pipeline huấn luyện end-to-end.
- So sánh nhiều cấu hình hyperparameter.
- Log metrics/artifacts vào MLflow.
- Đăng ký model tốt nhất lên Model Registry.
- Có DAG Airflow để chạy tự động theo lịch.

## Cấu trúc thư mục

```text
lab2-done/
├── pipeline/
│   ├── config.py
│   ├── data_ingestion.py
│   ├── preprocessing.py
│   ├── training.py
│   ├── evaluation.py
│   ├── registry.py
│   └── run_pipeline.py
├── experiments/
│   └── run_experiments.py
├── dags/
│   └── ml_training_dag.py
├── tests/
│   └── test_pipeline.py
├── models/
├── requirements.txt
└── pytest.ini
```

## Công nghệ chính

- Python 3.10+
- scikit-surprise
- MLflow
- Apache Airflow
- pytest

## Cài đặt

### 1) Tạo môi trường và cài package

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

### 2) Khởi chạy MLflow UI (tuỳ chọn)

```bash
mlflow ui --host 0.0.0.0 --port 5000
```

Mặc định trong code dùng:

- `MLFLOW_TRACKING_URI=http://localhost:5000`
- `MLFLOW_EXPERIMENT_NAME=movie-rating-prediction`

## Chạy pipeline

### Chạy nhanh với cấu hình mặc định

```bash
python -m pipeline.run_pipeline
```

### Chỉ định model/hyperparameters

```bash
python -m pipeline.run_pipeline --model-type svd --n-factors 100 --n-epochs 20
```

### Chạy và đăng ký model tốt nhất

```bash
python -m pipeline.run_pipeline --model-type svd --register
```

## Chạy experiments (hyperparameter tuning)

```bash
python -m experiments.run_experiments
```

Kết quả:

- Log toàn bộ run lên MLflow
- Tạo `experiment_report.md`
- In top run theo RMSE

## Airflow DAG

File DAG: `dags/ml_training_dag.py`

Luồng chính:

1. `load_data`
2. `preprocess_data`
3. `train_model`
4. `evaluate_model`
5. `decide_registration`
6. `register_model` hoặc `skip_registration`
7. `cleanup`

## Chạy test

```bash
pytest tests/ -v
```

Bỏ test nặng:

```bash
pytest tests/ -v -m "not slow"
```

## Biến môi trường quan trọng

- `MLFLOW_TRACKING_URI`
- `MLFLOW_EXPERIMENT_NAME`

Ví dụ PowerShell:

```powershell
$env:MLFLOW_TRACKING_URI="http://localhost:5000"
$env:MLFLOW_EXPERIMENT_NAME="movie-rating-prediction"
python -m pipeline.run_pipeline
```

## Ghi chú

- Dữ liệu dùng `ml-100k` (Surprise built-in dataset).
- Metrics chính: `rmse`, `mae`.
- Mô hình hỗ trợ: `svd`, `nmf`, `knn`.
