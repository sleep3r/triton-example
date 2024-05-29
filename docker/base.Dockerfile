FROM python:3.10-buster

WORKDIR /home/src

ARG POETRY_HOME=/etc/poetry
ENV PATH=${POETRY_HOME}/bin:${PATH}

RUN mkdir -p /home/src \
    && apt-get update || true \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        apt-utils software-properties-common build-essential bash-completion ca-certificates \
        htop vim gawk telnet tmux git tig screen openssh-client wget curl cmake unzip gcc \
        locales locales-all \
    && curl -sSL https://install.python-poetry.org | python3 - \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

ENV LC_ALL=en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US.UTF-8