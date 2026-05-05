"""Typed LSP client helpers built on the async stdio transport."""

from __future__ import annotations

import hashlib
from collections.abc import Awaitable, Sequence
from pathlib import Path
from typing import Protocol, TypeVar, cast
from urllib.parse import unquote, urlparse

from lsprotocol import types as lsp_types
from lsprotocol.converters import get_converter

from src.code_intel.core import (
    Diagnostic,
    DiagnosticSeverity,
    HoverInfo,
    Location,
    LSPTimeout,
    ProviderUnavailable,
    Range,
    Symbol,
    SymbolKind,
)
from src.code_intel.core.errors import CodeIntelError
from src.code_intel.core.models import validate_workspace_relative_path
from src.code_intel.tracing import result_count, trace_span

LSP_SYMBOL_SOURCE = "lsp"
LSP_SYMBOL_INDEX_VERSION = "lsp:document-symbol-v1"
LSP_SYMBOL_CONFIDENCE = 0.85

ModelT = TypeVar("ModelT")


class _LSPConverter(Protocol):
    def unstructure(self, obj: object) -> object: ...

    def structure(self, obj: object, cl: type[ModelT]) -> ModelT: ...


class _LSPNotificationHandler(Protocol):
    def __call__(self, params_payload: object | None, /) -> Awaitable[None]: ...


class _LSPTransport(Protocol):
    async def add_notification_handler(
        self, method: str, handler: _LSPNotificationHandler
    ) -> None: ...

    async def start(self) -> None: ...

    async def request(
        self, method: str, params: object | None = None, *, timeout: float | None = None
    ) -> object | None: ...

    async def notify(self, method: str, params: object | None = None) -> None: ...

    async def is_running(self) -> bool: ...

    async def close(self) -> None: ...


class LSPClient:
    """Small typed LSP client boundary for T11 transport primitives."""

    def __init__(
        self,
        transport: _LSPTransport,
        *,
        workspace_root: str | Path = ".",
        language: str = "unknown",
        source: str = LSP_SYMBOL_SOURCE,
    ) -> None:
        self._transport: _LSPTransport = transport
        self._workspace_root: Path = (
            Path(workspace_root).expanduser().resolve(strict=False)
        )
        self._language: str = language
        self._source: str = source
        self._converter: _LSPConverter = cast(_LSPConverter, get_converter())
        self._diagnostics_by_path: dict[str, list[Diagnostic]] = {}
        self._notification_handler_registered: bool = False

    async def start(self) -> None:
        if not self._notification_handler_registered:
            await self._transport.add_notification_handler(
                lsp_types.TEXT_DOCUMENT_PUBLISH_DIAGNOSTICS,
                self._handle_publish_diagnostics,
            )
            self._notification_handler_registered = True
        await self._transport.start()

    async def initialize(
        self,
        *,
        root_uri: str | None = None,
        initialization_options: object | None = None,
    ) -> lsp_types.InitializeResult:
        await self.start()
        params = lsp_types.InitializeParams(
            capabilities=lsp_types.ClientCapabilities(),
            process_id=None,
            client_info=lsp_types.ClientInfo(name="HPD-Agent CodeIntel", version="0.1"),
            root_uri=root_uri or self._workspace_root.as_uri(),
            initialization_options=initialization_options,
        )
        result = await self._request_typed(lsp_types.INITIALIZE, params)
        initialize_result = self._structure(
            result, lsp_types.InitializeResult, "invalid initialize response"
        )
        await self._transport.notify(lsp_types.INITIALIZED, {})
        return initialize_result

    async def did_open(
        self, path: str, *, language_id: str, text: str, version: int
    ) -> None:
        relative_path = self._validate_path(path)
        params = lsp_types.DidOpenTextDocumentParams(
            text_document=lsp_types.TextDocumentItem(
                uri=self._path_to_uri(relative_path),
                language_id=language_id,
                version=version,
                text=text,
            )
        )
        await self._notify_typed(lsp_types.TEXT_DOCUMENT_DID_OPEN, params)

    async def did_change(self, path: str, *, text: str, version: int) -> None:
        relative_path = self._validate_path(path)
        params = lsp_types.DidChangeTextDocumentParams(
            text_document=lsp_types.VersionedTextDocumentIdentifier(
                uri=self._path_to_uri(relative_path),
                version=version,
            ),
            content_changes=[
                lsp_types.TextDocumentContentChangeWholeDocument(text=text)
            ],
        )
        await self._notify_typed(lsp_types.TEXT_DOCUMENT_DID_CHANGE, params)

    async def did_save(self, path: str, *, text: str | None = None) -> None:
        relative_path = self._validate_path(path)
        params = lsp_types.DidSaveTextDocumentParams(
            text_document=lsp_types.TextDocumentIdentifier(
                uri=self._path_to_uri(relative_path)
            ),
            text=text,
        )
        await self._notify_typed(lsp_types.TEXT_DOCUMENT_DID_SAVE, params)

    async def goto_definition(
        self, path: str, *, line: int, character: int
    ) -> list[Location]:
        relative_path = self._validate_path(path)
        params = lsp_types.DefinitionParams(
            text_document=lsp_types.TextDocumentIdentifier(
                uri=self._path_to_uri(relative_path)
            ),
            position=lsp_types.Position(line=line, character=character),
        )
        result = await self._request_typed(lsp_types.TEXT_DOCUMENT_DEFINITION, params)
        return self._definition_locations(result)

    async def find_references(
        self,
        path: str,
        *,
        line: int,
        character: int,
        include_declaration: bool = True,
    ) -> list[Location]:
        relative_path = self._validate_path(path)
        params = lsp_types.ReferenceParams(
            text_document=lsp_types.TextDocumentIdentifier(
                uri=self._path_to_uri(relative_path)
            ),
            position=lsp_types.Position(line=line, character=character),
            context=lsp_types.ReferenceContext(include_declaration=include_declaration),
        )
        result = await self._request_typed(lsp_types.TEXT_DOCUMENT_REFERENCES, params)
        return self._reference_locations(result)

    async def hover(self, path: str, *, line: int, character: int) -> HoverInfo | None:
        relative_path = self._validate_path(path)
        params = lsp_types.HoverParams(
            text_document=lsp_types.TextDocumentIdentifier(
                uri=self._path_to_uri(relative_path)
            ),
            position=lsp_types.Position(line=line, character=character),
        )
        result = await self._request_typed(lsp_types.TEXT_DOCUMENT_HOVER, params)
        if result is None:
            return None
        if not isinstance(result, dict):
            raise ProviderUnavailable("invalid hover response")
        hover_payload = cast(dict[str, object], result)
        hover = self._structure(
            hover_payload, lsp_types.Hover, "invalid hover response"
        )
        return HoverInfo(
            contents=self._hover_contents_to_text(hover.contents),
            range=None if hover.range is None else self._core_range(hover.range),
            source=self._source,
        )

    async def document_symbols(self, path: str) -> list[Symbol]:
        await self.start()
        relative_path = self._validate_path(path)
        params = lsp_types.DocumentSymbolParams(
            text_document=lsp_types.TextDocumentIdentifier(
                uri=self._path_to_uri(relative_path)
            )
        )
        result = await self._request_typed(
            lsp_types.TEXT_DOCUMENT_DOCUMENT_SYMBOL, params
        )
        if result is None:
            return []
        if not isinstance(result, list):
            raise ProviderUnavailable("invalid document symbol response")
        symbols: list[Symbol] = []
        items = cast(list[object], result)
        for item in items:
            if not isinstance(item, dict):
                raise ProviderUnavailable("invalid document symbol item")
            item_payload = cast(dict[str, object], item)
            if "location" in item_payload:
                symbol_information = self._structure(
                    item_payload,
                    lsp_types.SymbolInformation,
                    "invalid symbol information response",
                )
                converted = self._symbol_from_information(symbol_information)
                if converted is not None:
                    symbols.append(converted)
                continue
            document_symbol = self._structure(
                item_payload,
                lsp_types.DocumentSymbol,
                "invalid document symbol response",
            )
            symbols.extend(
                self._symbols_from_document_symbol(
                    document_symbol, relative_path, None, None
                )
            )
        return symbols

    async def diagnostics(self, path: str) -> list[Diagnostic]:
        relative_path = self._validate_path(path)
        return list(self._diagnostics_by_path.get(relative_path, ()))

    async def shutdown(self) -> None:
        if not await self._transport.is_running():
            await self.close()
            return
        try:
            _ = await self._transport.request(lsp_types.SHUTDOWN, None)
            await self._transport.notify(lsp_types.EXIT, None)
        except CodeIntelError:
            pass
        finally:
            await self.close()

    async def close(self) -> None:
        await self._transport.close()

    async def is_running(self) -> bool:
        return await self._transport.is_running()

    async def _request_typed(
        self, method: str, params: object | None = None
    ) -> object | None:
        with trace_span(
            f"lsp.{self._source}.{method}",
            {"provider_name": self._source, "language": self._language},
        ) as span:
            try:
                payload = (
                    None if params is None else self._converter.unstructure(params)
                )
                result = await self._transport.request(method, payload)
                span.add_metadata({"result_count": result_count(result)})
                return result
            except (LSPTimeout, ProviderUnavailable):
                raise
            except CodeIntelError:
                raise
            except Exception:
                raise ProviderUnavailable("LSP client request failed") from None

    async def _notify_typed(self, method: str, params: object | None = None) -> None:
        with trace_span(
            f"lsp.{self._source}.{method}",
            {"provider_name": self._source, "language": self._language},
        ) as span:
            try:
                payload = (
                    None if params is None else self._converter.unstructure(params)
                )
                await self._transport.notify(method, payload)
                span.add_metadata({"result_count": 0})
            except ProviderUnavailable:
                raise
            except CodeIntelError:
                raise
            except Exception:
                raise ProviderUnavailable("LSP client notification failed") from None

    async def _handle_publish_diagnostics(self, params_payload: object | None) -> None:
        if params_payload is None:
            return
        try:
            params = self._structure(
                params_payload,
                lsp_types.PublishDiagnosticsParams,
                "invalid publish diagnostics notification",
            )
        except ProviderUnavailable:
            return
        path = self._path_from_uri(params.uri)
        if path is None:
            return
        self._diagnostics_by_path[path] = [
            self._core_diagnostic(path, diagnostic) for diagnostic in params.diagnostics
        ]

    def _definition_locations(self, payload: object | None) -> list[Location]:
        if payload is None:
            return []
        items = cast(list[object], payload) if isinstance(payload, list) else [payload]
        locations: list[Location] = []
        for item in items:
            if not isinstance(item, dict):
                raise ProviderUnavailable("invalid definition response")
            item_payload = cast(dict[str, object], item)
            if "targetUri" in item_payload:
                link = self._structure(
                    item_payload,
                    lsp_types.LocationLink,
                    "invalid definition location link",
                )
                location = self._core_location_from_link(link)
            else:
                lsp_location = self._structure(
                    item_payload, lsp_types.Location, "invalid definition location"
                )
                location = self._core_location_from_lsp(lsp_location)
            if location is not None:
                locations.append(location)
        return locations

    def _reference_locations(self, payload: object | None) -> list[Location]:
        if payload is None:
            return []
        if not isinstance(payload, list):
            raise ProviderUnavailable("invalid references response")
        locations: list[Location] = []
        for item in cast(list[object], payload):
            if not isinstance(item, dict):
                raise ProviderUnavailable("invalid reference location")
            lsp_location = self._structure(
                cast(dict[str, object], item),
                lsp_types.Location,
                "invalid reference location",
            )
            location = self._core_location_from_lsp(lsp_location)
            if location is not None:
                locations.append(location)
        return locations

    def _core_location_from_lsp(self, location: lsp_types.Location) -> Location | None:
        path = self._path_from_uri(location.uri)
        if path is None:
            return None
        return Location(path=path, range=self._core_range(location.range))

    def _core_location_from_link(self, link: lsp_types.LocationLink) -> Location | None:
        path = self._path_from_uri(link.target_uri)
        if path is None:
            return None
        return Location(path=path, range=self._core_range(link.target_selection_range))

    def _structure(
        self, payload: object, model_type: type[ModelT], detail: str
    ) -> ModelT:
        try:
            return self._converter.structure(payload, model_type)
        except Exception:
            raise ProviderUnavailable(detail) from None

    def _symbols_from_document_symbol(
        self,
        symbol: lsp_types.DocumentSymbol,
        path: str,
        parent_id: str | None,
        parent_qualified_name: str | None,
    ) -> list[Symbol]:
        qualified_name = (
            f"{parent_qualified_name}.{symbol.name}"
            if parent_qualified_name
            else symbol.name
        )
        core_symbol = Symbol(
            name=symbol.name,
            qualified_name=qualified_name,
            kind=self._symbol_kind(symbol.kind),
            language=self._language,
            path=path,
            range=self._core_range(symbol.range),
            selection_range=self._core_range(symbol.selection_range),
            parent_id=parent_id,
            signature=symbol.detail,
            doc=None,
            source=self._source,
            confidence=LSP_SYMBOL_CONFIDENCE,
            file_hash="lsp-untracked",
            index_version=LSP_SYMBOL_INDEX_VERSION,
        )
        symbols = [core_symbol]
        children: Sequence[lsp_types.DocumentSymbol] = symbol.children or ()
        for child in children:
            symbols.extend(
                self._symbols_from_document_symbol(
                    child, path, core_symbol.id, qualified_name
                )
            )
        return symbols

    def _symbol_from_information(
        self, symbol: lsp_types.SymbolInformation
    ) -> Symbol | None:
        path = self._path_from_uri(symbol.location.uri)
        if path is None:
            return None
        qualified_name = (
            f"{symbol.container_name}.{symbol.name}"
            if symbol.container_name
            else symbol.name
        )
        return Symbol(
            name=symbol.name,
            qualified_name=qualified_name,
            kind=self._symbol_kind(symbol.kind),
            language=self._language,
            path=path,
            range=self._core_range(symbol.location.range),
            selection_range=None,
            parent_id=None,
            signature=None,
            doc=None,
            source=self._source,
            confidence=LSP_SYMBOL_CONFIDENCE,
            file_hash="lsp-untracked",
            index_version=LSP_SYMBOL_INDEX_VERSION,
        )

    def _core_diagnostic(
        self, path: str, diagnostic: lsp_types.Diagnostic
    ) -> Diagnostic:
        severity = self._diagnostic_severity(diagnostic.severity)
        code = None if diagnostic.code is None else str(diagnostic.code)
        source = diagnostic.source or self._source
        core_range = self._core_range(diagnostic.range)
        fingerprint = self._diagnostic_fingerprint(
            path=path,
            source=source,
            severity=severity,
            code=code,
            message=diagnostic.message,
            diagnostic_range=core_range,
        )
        return Diagnostic(
            path=path,
            range=core_range,
            severity=severity,
            message=diagnostic.message,
            code=code,
            source=source,
            fingerprint=fingerprint,
        )

    @staticmethod
    def _core_range(lsp_range: lsp_types.Range) -> Range:
        return Range(
            start_line=lsp_range.start.line,
            start_col=lsp_range.start.character,
            end_line=lsp_range.end.line,
            end_col=lsp_range.end.character,
        )

    @staticmethod
    def _symbol_kind(kind: lsp_types.SymbolKind) -> SymbolKind:
        mapping = {
            lsp_types.SymbolKind.Module: SymbolKind.MODULE,
            lsp_types.SymbolKind.Namespace: SymbolKind.NAMESPACE,
            lsp_types.SymbolKind.Class: SymbolKind.CLASS,
            lsp_types.SymbolKind.Interface: SymbolKind.INTERFACE,
            lsp_types.SymbolKind.Enum: SymbolKind.ENUM,
            lsp_types.SymbolKind.EnumMember: SymbolKind.ENUM_MEMBER,
            lsp_types.SymbolKind.Method: SymbolKind.METHOD,
            lsp_types.SymbolKind.Function: SymbolKind.FUNCTION,
            lsp_types.SymbolKind.Variable: SymbolKind.VARIABLE,
            lsp_types.SymbolKind.Constant: SymbolKind.VARIABLE,
            lsp_types.SymbolKind.Field: SymbolKind.VARIABLE,
            lsp_types.SymbolKind.Property: SymbolKind.VARIABLE,
            lsp_types.SymbolKind.File: SymbolKind.MODULE,
        }
        return mapping.get(kind, SymbolKind.VARIABLE)

    @staticmethod
    def _diagnostic_severity(
        severity: lsp_types.DiagnosticSeverity | None,
    ) -> DiagnosticSeverity:
        if severity == lsp_types.DiagnosticSeverity.Error:
            return DiagnosticSeverity.ERROR
        if severity == lsp_types.DiagnosticSeverity.Warning:
            return DiagnosticSeverity.WARNING
        if severity == lsp_types.DiagnosticSeverity.Hint:
            return DiagnosticSeverity.HINT
        return DiagnosticSeverity.INFO

    @staticmethod
    def _diagnostic_fingerprint(
        *,
        path: str,
        source: str,
        severity: DiagnosticSeverity,
        code: str | None,
        message: str,
        diagnostic_range: Range,
    ) -> str:
        raw = (
            f"{source}:{path}:{diagnostic_range.start_line}:{diagnostic_range.start_col}:"
            f"{diagnostic_range.end_line}:{diagnostic_range.end_col}:{severity.value}:{code or ''}:{message}"
        )
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]

    @classmethod
    def _hover_contents_to_text(cls, contents: object) -> str:
        if isinstance(contents, str):
            return contents
        if isinstance(contents, lsp_types.MarkupContent):
            return contents.value
        if isinstance(contents, lsp_types.MarkedStringWithLanguage):
            return contents.value
        if isinstance(contents, Sequence):
            parts = [cls._hover_contents_to_text(item) for item in contents]
            return "\n\n".join(part for part in parts if part)
        return str(contents)

    def _path_to_uri(self, path: str) -> str:
        return (self._workspace_root / path).resolve(strict=False).as_uri()

    def _path_from_uri(self, uri: str) -> str | None:
        parsed = urlparse(uri)
        if parsed.scheme != "file":
            return None
        candidate = Path(unquote(parsed.path)).resolve(strict=False)
        try:
            relative = candidate.relative_to(self._workspace_root).as_posix()
        except ValueError:
            return None
        if not relative or relative == ".":
            return None
        try:
            return validate_workspace_relative_path(relative)
        except ValueError:
            return None

    @staticmethod
    def _validate_path(path: str) -> str:
        try:
            return validate_workspace_relative_path(path)
        except ValueError:
            raise ProviderUnavailable("invalid LSP workspace path") from None


__all__ = [
    "LSPClient",
    "LSP_SYMBOL_CONFIDENCE",
    "LSP_SYMBOL_INDEX_VERSION",
    "LSP_SYMBOL_SOURCE",
]
