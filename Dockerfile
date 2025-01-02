FROM python:3.10-buster

# Install Poetry
RUN pip install poetry

# Set Poetry environment variables
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# Set the working directory
WORKDIR /app

# Copy dependency
COPY pyproject.toml poetry.lock ./
RUN poetry install --without dev --no-root && rm -rf "$POETRY_CACHE_DIR"

# Copy the entire repository
COPY . .

# Final installation step (in case your project includes additional dependencies)
RUN poetry install --without dev

# Default command can be overridden by Docker Compose
ENTRYPOINT ["poetry", "run", "python"]
