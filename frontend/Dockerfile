# frontend/Dockerfile

FROM --platform=linux/amd64 python:3.10

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# EXPOSE 8501

# CMD ["streamlit", "run", "main.py"]

CMD streamlit run --server.port $PORT main.py 

