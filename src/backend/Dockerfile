FROM python:3.12-slim

# Install curl for healthchecks
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Set environment variables
ENV DOCKER_ENV=true

# Expose ports for the services
EXPOSE 7821 4321 3210

# Command to run the services
CMD ["bash", "run_services.sh"]
