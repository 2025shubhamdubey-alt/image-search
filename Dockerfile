FROM python:3.10-slim

# Set working directory
WORKDIR /

# Copy and Install python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Expose both Streamlit and FastAPI ports
EXPOSE 8501 8000

CMD ["python", "run.py"]
