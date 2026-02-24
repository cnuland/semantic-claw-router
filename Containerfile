# ── Semantic Claw Router ─────────────────────────────────────────────
# Multi-stage build for minimal production image.
#
# Build variants:
#   podman build -t semantic-claw-router .                      # base (no ML)
#   podman build --build-arg INSTALL_SEMANTIC=1 -t semantic-claw-router .  # with semantic classifier
#
# Run:
#   podman run -p 8080:8080 -v ./config.yaml:/app/config.yaml:ro \
#     -e GEMINI_API_KEY=... -e VLLM_ENDPOINT=... \
#     semantic-claw-router --config /app/config.yaml
#
# Push to quay.io:
#   podman tag semantic-claw-router quay.io/<org>/semantic-claw-router:latest
#   podman push quay.io/<org>/semantic-claw-router:latest

# ── Stage 1: Build ──────────────────────────────────────────────────
FROM registry.access.redhat.com/ubi9/python-311:latest AS builder

USER 0
WORKDIR /build

# Copy project files
COPY pyproject.toml README.md LICENSE ./
COPY src/ src/

# Build arg: set to "1" to include sentence-transformers (adds ~500MB)
ARG INSTALL_SEMANTIC=0

# Install the package
RUN if [ "$INSTALL_SEMANTIC" = "1" ]; then \
      pip install --no-cache-dir --prefix=/install ".[semantic]"; \
    else \
      pip install --no-cache-dir --prefix=/install .; \
    fi

# ── Stage 2: Runtime ───────────────────────────────────────────────
FROM registry.access.redhat.com/ubi9/python-311:latest AS runtime

LABEL org.opencontainers.image.title="Semantic Claw Router" \
      org.opencontainers.image.description="Intelligent LLM request router — mixture-of-models at the system level" \
      org.opencontainers.image.source="https://github.com/cnuland/semantic-claw-router" \
      org.opencontainers.image.licenses="Apache-2.0"

WORKDIR /app

# Copy installed packages from builder into UBI9's site-packages path
COPY --from=builder /install/lib /opt/app-root/lib
COPY --from=builder /install/bin /opt/app-root/bin

# Copy examples (useful for reference, not required)
COPY examples/ examples/

# Ensure /opt/app-root/bin is in PATH (UBI9 default includes it)
ENV PATH="/opt/app-root/bin:${PATH}"

# UBI images run as non-root by default (uid 1001)
USER 1001

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

ENTRYPOINT ["semantic-claw-router"]
CMD ["--config", "/app/config.yaml", "--port", "8080"]
