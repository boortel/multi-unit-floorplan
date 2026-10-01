FROM ravers/python_ml:v0

ENV PATH="/opt/miniconda3/bin:${PATH}"

RUN apt-get update && apt-get install -y git curl \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get remove -y libnode-dev libnode72 2>/dev/null || true \
    && apt-get install -y nodejs \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
