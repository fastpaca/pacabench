FROM python:3.12-slim

# Install git (needed by gitpython) and uv
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Set working directory
WORKDIR /app

# Copy dependency files first for caching
COPY pyproject.toml uv.lock ./

# Install dependencies
# We use --system to install into the system python, keeping the image small
# and avoiding virtualenv overhead inside the container
RUN uv sync --frozen --no-dev

# Copy the rest of the application
COPY pacabench/ pacabench/

# Set python path
ENV PYTHONPATH=/app

# Default entrypoint
ENTRYPOINT ["uv", "run", "pacabench"]

