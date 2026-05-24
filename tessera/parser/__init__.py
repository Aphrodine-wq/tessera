"""Parser: .t.md → ParsedModule."""

from .module import ParsedModule, SubstrateBlock, parse_file, parse_source

__all__ = ["ParsedModule", "SubstrateBlock", "parse_file", "parse_source"]
