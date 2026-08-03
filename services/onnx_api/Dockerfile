FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV MODEL_ROOT=/app/models/goal15-final-v1

COPY services/onnx_api/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY src/multisensor_ml/__init__.py /app/multisensor_ml/__init__.py
COPY src/multisensor_ml/dual_sensor_onnx.py /app/multisensor_ml/dual_sensor_onnx.py
COPY src/multisensor_ml/cloud_run_onnx.py /app/multisensor_ml/cloud_run_onnx.py
COPY services/onnx_api/app.py /app/app.py
COPY services/onnx_api/models/goal15-final-v1 /app/models/goal15-final-v1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
