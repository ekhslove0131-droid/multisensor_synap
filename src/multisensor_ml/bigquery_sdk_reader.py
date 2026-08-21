"""Optional Google BigQuery SDK execution boundary for frozen cohort queries."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from typing import Protocol, cast


class BigQuerySchemaFieldProtocol(Protocol):
    @property
    def name(self) -> str: ...


class BigQueryRowIteratorProtocol(Protocol):
    @property
    def schema(self) -> Sequence[BigQuerySchemaFieldProtocol]: ...

    def __iter__(self) -> Iterator[object]: ...


class BigQueryQueryJobProtocol(Protocol):
    def result(self) -> BigQueryRowIteratorProtocol: ...


class BigQueryClientProtocol(Protocol):
    def query(
        self,
        query: str,
        *,
        job_config: object,
        project: str,
        location: str,
    ) -> BigQueryQueryJobProtocol: ...


class _ScalarQueryParameterFactory(Protocol):
    def __call__(self, name: str, parameter_type: str, value: str) -> object: ...


class _QueryJobConfigFactory(Protocol):
    def __call__(self, *, query_parameters: Sequence[object]) -> object: ...


class _BigQueryModuleProtocol(Protocol):
    ScalarQueryParameter: _ScalarQueryParameterFactory
    QueryJobConfig: _QueryJobConfigFactory


QueryJobConfigFactory = Callable[[str, str, str], object]


@dataclass(frozen=True, slots=True)
class SdkQueryRows:
    schema_fields: tuple[str, ...]
    rows: tuple[dict[str, object], ...]


def _default_job_config(
    parameter_name: str,
    parameter_type: str,
    parameter_value: str,
) -> object:
    try:
        module = cast(
            _BigQueryModuleProtocol, import_module("google.cloud.bigquery")
        )
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "google-cloud-bigquery is required only when executing the SDK reader"
        ) from error
    return module.QueryJobConfig(
        query_parameters=[
            module.ScalarQueryParameter(
                parameter_name, parameter_type, parameter_value
            )
        ]
    )


def _row_mapping(row: object) -> dict[str, object]:
    if isinstance(row, Mapping):
        return {str(key): value for key, value in row.items()}
    items = getattr(row, "items", None)
    if not callable(items):
        raise ValueError("BigQuery SDK row must be a mapping or Row-like object")
    try:
        return {str(key): value for key, value in items()}
    except (TypeError, ValueError) as error:
        raise ValueError(
            "BigQuery SDK row must be a mapping or Row-like object"
        ) from error


def execute_read_only_sdk_query(
    *,
    client: BigQueryClientProtocol,
    query_contract: Mapping[str, object],
    project: str,
    location: str,
    job_config_factory: QueryJobConfigFactory | None = None,
) -> SdkQueryRows:
    """Execute one parameterized authorized-view query through the Google SDK."""

    if query_contract.get("locked_access") is not False:
        raise ValueError("SDK query contract must forbid LOCKED access")
    transports = query_contract.get("allowed_transports")
    if not isinstance(transports, Sequence) or isinstance(transports, (str, bytes)):
        raise ValueError("SDK query contract has invalid transports")
    if "google_cloud_bigquery_sdk" not in transports:
        raise ValueError("SDK query contract does not authorize Google BigQuery SDK")
    query = query_contract.get("query")
    if not isinstance(query, str) or not query:
        raise ValueError("SDK query contract has invalid query")
    parameter = query_contract.get("parameter")
    if not isinstance(parameter, Mapping) or set(parameter) != {"name", "type", "value"}:
        raise ValueError("SDK query contract has invalid parameter")
    name = parameter["name"]
    parameter_type = parameter["type"]
    value = parameter["value"]
    if not all(isinstance(item, str) and item for item in (name, parameter_type, value)):
        raise ValueError("SDK query contract parameter must contain strings")
    if parameter_type != "STRING":
        raise ValueError("SDK query contract parameter must be STRING")

    factory = job_config_factory or _default_job_config
    job_config = factory(name, parameter_type, value)
    iterator = client.query(
        query,
        job_config=job_config,
        project=project,
        location=location,
    ).result()
    schema_fields: list[str] = []
    for field in iterator.schema:
        field_name = getattr(field, "name", None)
        if not isinstance(field_name, str) or not field_name:
            raise ValueError("BigQuery SDK schema field is invalid")
        schema_fields.append(field_name)
    return SdkQueryRows(
        schema_fields=tuple(schema_fields),
        rows=tuple(_row_mapping(row) for row in iterator),
    )
