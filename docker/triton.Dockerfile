FROM nvcr.io/nvidia/tritonserver:22.09-py3
ARG MODELS_FOLDER

ENV LC_ALL=en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US.UTF-8

COPY ${MODELS_FOLDER} /models

CMD ["tritonserver", "--model-repository=/models"]