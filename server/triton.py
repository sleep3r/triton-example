import ssl
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Type, Union
from urllib.parse import ParseResult, urlparse

import numpy as np
import tritonclient.grpc as triton_grpc
import tritonclient.http as triton_http
from loguru import logger


@dataclass
class TritonClientSettings:
    """
    Setting for Triton Clients.

    Supports both http(s) and grpc.

    Parameters
    ----------
    url: str
        The inference server name, port and optional base path
        in the following format: scheme://host[:port]/<base-path>,
        e.g. http://localhost:8000, grpc://triton:443.
    model: str
        The name of the model to check for readiness.
    version: str
        The version of the model to check for readiness. The default value
        is an empty string which means then the server will choose a version
        based on the model and internal policy.
    secure: bool
        If True, channels the requests to encrypted https scheme.
        Some improper settings may cause connection to prematurely
        terminate with an unsuccessful handshake. See
        `ssl_context_factory` option for using secure default
        settings. It is always True for https connection. Defaults to False.
    cert_path: Optional[Path]
        File holding the PEM-encoded root certificates as a byte
        string, or None to retrieve them from a default location
        chosen by gRPC runtime. The option is ignored if `ssl`
        is False. Defaults to None.
    verbose: bool
        If True generate verbose output. Defaults to False.
    concurrency: int
        The number of connections to create for this client.
        Defaults to 1.
    connection_timeout: float
        The timeout value for the connection. Defaults to 60.0.
    network_timeout: float
        The timeout value for the network. Defaults to 60.0.
    max_greenlets: Optional[int]
        Determines the maximum allowed number of worker greenlets
        for handling asynchronous inference requests. Defaults to
        is None, which means there will be no restriction on the
        number of greenlets created.
    """

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
        """
        Get InferenceClient for the scheme.

        Returns
        -------
        Union[triton_http.InferenceServerClient, triton_grpc.InferenceServerClient]
            InfernceSeverClient depending on the protocol.

        Raises
        ------
        ValueError
            If the scheme is not http(s) or grpc.
        """
        if self.parsed_url.scheme.startswith("http"):
            ssl_context_factory = (
                lambda: ssl.create_default_context(
                    cafile=self.cert_path
                )  # pylint: disable=C3001
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
        """
        Get InferInput for the scheme.

        Returns
        -------
        Union[triton_http.InferInput, triton_grpc.InferInput]
            InferInput depending on the protocol.

        Raises
        ------
        ValueError
            If the scheme is not http(s) or grpc.
        """
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
        """
        Get InferRequestedOutput for the scheme.

        Returns
        -------
        Union[triton_http.InferRequestedOutput, triton_grpc.InferRequestedOutput]
            InferRequestedOutput depending on the protocol.

        Raises
        ------
        ValueError
            If the scheme is not http(s) or grpc.
        """
        if self.parsed_url.scheme.startswith("http"):
            return triton_http.InferRequestedOutput
        if self.parsed_url.scheme == "grpc":
            return triton_grpc.InferRequestedOutput
        raise ValueError(
            f"Invalid scheme {self.parsed_url.scheme}. Supported values: http, https, grpc."
        )


class TritonClient:
    """
    Client is used to perform any kind of communication
    with the Triton Server using http/grpc protocol (depends on settings).

    Parameters
    ----------
    settings: TritonClientSettings
        Settings to configure Triton Server connection.
    output_keys: list[str]
        Output keys to get from the model.
    """

    def __init__(self, settings: TritonClientSettings, output_keys: list[str]) -> None:
        self.settings = settings
        self._client = settings.get_inference_client()
        self._infer_input = settings.get_infer_input()
        self._infer_output = settings.get_infer_requested_output()
        self._output_keys = output_keys

    def _prepare_inputs(self, inputs: dict[str, np.ndarray]):
        infer_inputs = {
            k: self._infer_input(k, [len(v), v[0].shape[-1]], "INT64")
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
        """
        Get predictions from the model for the `inputs`.

        You need to call `as_numpy` function on `predict` output to get predictions as a numpy array.

        Parameters
        ----------
        inputs: dict[str, np.ndarray]
            Dictionary of keys required for the model and tensor inputs.

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary of predictions for `output_keys`.
        """
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
    ) -> tuple[Optional[str], bool]:  # pylint: disable=too-many-return-statements
        """
        Whether the server is ready for inference.

        Returns
        -------
        tuple[Optional[str], bool]
            Returns a tuple of 2 values:
            - An error message in case server is not ready. If the server is ready message is None.
            - Whether server is ready boolean value.
        """
        if not self._client.is_server_live():
            return "is_server_live failed", False
        if not self._client.is_server_ready():
            return "is_server_ready failed", False
        if not self._client.is_model_ready(self.settings.model):
            return "is_model_ready failed", False
        return None, True
