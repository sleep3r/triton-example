ARG BASE_IMAGE

# Configure dependencies
FROM ${BASE_IMAGE} as venv
WORKDIR /home/src
COPY poetry.lock pyproject.toml Makefile /home/src/
RUN make install

# Production build
FROM ${BASE_IMAGE} as api
WORKDIR /home/src
# Enviroment variables to run application
ENV PORT=8000
ENV WORKERS=1
# Set directory to store Prometheus Metrics
ENV PROMETHEUS_MULTIPROC_DIR=/metrics
RUN mkdir -p /metrics
# Copy everytihng we need
COPY --from=venv /home/src/.venv .venv
COPY . .
CMD ["make", "fastapi"]