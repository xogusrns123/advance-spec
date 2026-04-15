"""BFCL multi-turn tool utilities.

Provides DuckDuckGo search monkey-patching for WebSearchAPI,
replacing the paid SerpAPI. Patches the CLASS before any instances
are created, so execute_multi_turn_func_call() uses DuckDuckGo
from the first call.
"""

import re

from bfcl_eval.eval_checker.multi_turn_eval import multi_turn_utils as _mt_utils

_CLASS_PATCHED = False


def _ddg_search_engine_query(
    self,
    keywords: str,
    max_results: int = 10,
    region: str = "wt-wt",
) -> list | dict:
    """Replace SerpAPI with free duckduckgo-search (no API key required)."""
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS

        results = list(DDGS().text(keywords, region=region, max_results=max_results))

        filtered_results = [
            {
                "title": r.get("title", ""),
                "href": r.get("href", ""),
                "body": r.get("body", ""),
            }
            for r in results
        ]

        if hasattr(self, "show_snippet") and not self.show_snippet:
            filtered_results = [
                {"title": r["title"], "href": r["href"]} for r in filtered_results
            ]

        return filtered_results
    except Exception as e:
        return {"error": str(e)}


def patch_websearch_class():
    """Patch WebSearchAPI class to use DuckDuckGo BEFORE any instances are created.

    Must be called once at startup, before execute_multi_turn_func_call().
    """
    global _CLASS_PATCHED
    if _CLASS_PATCHED:
        return

    from bfcl_eval.eval_checker.multi_turn_eval.func_source_code.web_search import (
        WebSearchAPI,
    )
    WebSearchAPI.search_engine_query = _ddg_search_engine_query
    _CLASS_PATCHED = True


def patch_websearch_in_globals(entry_id: str):
    """Patch any already-created WebSearchAPI instances (backward compat)."""
    import types
    sanitized_id = re.sub(r"[-./:]", "_", entry_id)
    mt_globals = vars(_mt_utils)
    for key, val in list(mt_globals.items()):
        if "WebSearchAPI" in key and sanitized_id in key:
            val.search_engine_query = types.MethodType(_ddg_search_engine_query, val)


def cleanup_globals(entry_id: str):
    """Remove class instances from multi_turn_utils globals()."""
    sanitized_id = re.sub(r"[-./:]", "_", entry_id)
    mt_globals = vars(_mt_utils)
    keys_to_remove = [
        key for key in mt_globals if sanitized_id in key and "_instance" in key
    ]
    for key in keys_to_remove:
        del mt_globals[key]
