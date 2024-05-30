from dataclasses import dataclass, field
from pathlib import Path
import ssl
from typing import Optional, Type, Union
from urllib.parse import ParseResult, urlparse

from loguru import logger
import numpy as np
import tritonclient.grpc as triton_grpc
import tritonclient.http as triton_http


@dataclass
class TritonClientSettings:
    url: str
    model: str
    version: str
    secure: bool = False
    cert_path: Optional[Path] = None
    verbose: bool = False
    concurrency: int = 1
    connection_timeout: float = 60.0
    network_timeout: float = 60.0
    max_greenlets: Optional[int] = None
    parsed_url: ParseResult = field(init=False)

    def __post_init__(self) -> None:
        self.parsed_url = urlparse(self.url)
        if not self.secure and self.parsed_url.scheme == "https":
            self.secure = True

    def get_inference_client(
        self,
    ) -> Union[triton_http.InferenceServerClient, triton_grpc.InferenceServerClient]:
        if self.parsed_url.scheme.startswith("http"):
            ssl_context_factory = (
                lambda: ssl.create_default_context(
                    cafile=self.cert_path
                )
                if self.secure
                else None
            )
            return triton_http.InferenceServerClient(
                url=self.parsed_url.netloc,
                verbose=self.verbose,
                concurrency=self.concurrency,
                connection_timeout=self.connection_timeout,
                network_timeout=self.network_timeout,
                max_greenlets=self.max_greenlets,
                ssl=self.secure,
                ssl_context_factory=ssl_context_factory,
            )
        if self.parsed_url.scheme == "grpc":
            return triton_grpc.InferenceServerClient(
                url=self.parsed_url.netloc,
                verbose=self.verbose,
                ssl=self.secure,
                root_certificates=self.cert_path,
            )
        raise ValueError(
            f"Invalid scheme {self.parsed_url.scheme}. Supported values: http, https, grpc."
        )

    def get_infer_input(
        self,
    ) -> Union[Type[triton_http.InferInput], Type[triton_grpc.InferInput]]:
        if self.parsed_url.scheme.startswith("http"):
            return triton_http.InferInput
        if self.parsed_url.scheme == "grpc":
            return triton_grpc.InferInput
        raise ValueError(
            f"Invalid scheme {self.parsed_url.scheme}. Supported values: http, https, grpc."
        )

    def get_infer_requested_output(
        self,
    ) -> Union[
        Type[triton_http.InferRequestedOutput], Type[triton_grpc.InferRequestedOutput]
    ]:
        if self.parsed_url.scheme.startswith("http"):
            return triton_http.InferRequestedOutput
        if self.parsed_url.scheme == "grpc":
            return triton_grpc.InferRequestedOutput
        raise ValueError(
            f"Invalid scheme {self.parsed_url.scheme}. Supported values: http, https, grpc."
        )


class TritonClient:
    def __init__(self, settings: TritonClientSettings, output_keys: list[str]) -> None:
        self.settings = settings
        self._client = settings.get_inference_client()
        self._infer_input = settings.get_infer_input()
        self._infer_output = settings.get_infer_requested_output()
        self._output_keys = output_keys

    def _prepare_inputs(self, inputs: dict[str, np.ndarray]):
        infer_inputs = {
            k: self._infer_input(k, v.shape, "FP32")
            for k, v in inputs.items()
        }
        for k, v in inputs.items():
            infer_inputs[k].set_data_from_numpy(v)
        return infer_inputs.values()

    def _get_outputs(
        self,
    ) -> Union[
        list[triton_http.InferRequestedOutput], list[triton_grpc.InferRequestedOutput]
    ]:
        return [self._infer_output(x) for x in self._output_keys]

    def predict(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        infer_inputs = self._prepare_inputs(inputs)
        infer_outputs = self._get_outputs()
        try:
            results = self._client.infer(
                model_name=self.settings.model,
                model_version=self.settings.version,
                inputs=infer_inputs,
                outputs=infer_outputs,
            )
        except (
            triton_http.InferenceServerException,
            triton_grpc.InferenceServerException,
        ) as e:
            logger.error(f"Request to triton server failed: {repr(e)}", exc_info=True)
            return {}
        return {key: results.as_numpy(key) for key in self._output_keys}

    def ping(
        self,
    ) -> tuple[Optional[str], bool]:
        if not self._client.is_server_live():
            return "is_server_live failed", False
        if not self._client.is_server_ready():
            return "is_server_ready failed", False
        if not self._client.is_model_ready(self.settings.model):
            return "is_model_ready failed", False
        return None, True