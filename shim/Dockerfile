FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY shim.py .
EXPOSE 8787
ENV SHIM_HOST=0.0.0.0
CMD ["python", "shim.py"]
