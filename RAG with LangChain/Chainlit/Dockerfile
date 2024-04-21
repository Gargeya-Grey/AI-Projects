FROM python:3.9-slim

ENV PYTHONUNBUFFERED 1

WORKDIR /app

COPY requirements.txt /app

RUN pip install -r /app/requirements.txt

COPY . /app

EXPOSE 8080

CMD ["python", "-m", "chainlit", "run", "app.py", "-h", "--port", "8080"]