# Dockerfile

# Use a specific, stable version of the Debian OS (Bullseye)
FROM python:3.11-slim-bullseye

# Set this environment variable to prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /code

# Install the required system library using the new package name 'libgl1'
RUN apt-get update && \
    apt-get install -y libgl1 && \
    rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./app /code/app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]