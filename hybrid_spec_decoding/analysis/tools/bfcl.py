"""BFCL multi-turn tool utilities.

Provides DuckDuckGo search monkey-patching for WebSearchAPI instances
created by execute_multi_turn_func_call(), replacing the paid SerpAPI.

Note: execute_multi_turn_func_call() stores class instances in the
multi_turn_utils module's globals(), so we operate on that module's
namespace rather than this module's.
"""

import re
import types

from bfcl_eval.eval_checker.multi_turn_eval import multi_turn_utils as _mt_utils


def _ddg_search_engine_query(
    self,
    keywords: str,
    max_results: int = 10,
    region: str = "wt-wt",
) -> list | dict:
    """Replace SerpAPI with free duckduckgo-search (no API key required).

    Args:
        keywords: Search query string
        max_results: Number of results to return
        region: DuckDuckGo region code (e.g., "wt-wt", "us-en")

    Returns:
        List of dicts with "title", "href", "body" keys, or error dict.
    """
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS

        results = list(DDGS().text(keywords, region=region, max_results=max_results))

        # Filter to only include "title", "href", "body" keys to match expected format
        filtered_results = [
            {
                "title": r.get("title", ""),
                "href": r.get("href", ""),
                "body": r.get("body", ""),
            }
            for r in results
        ]

        # If show_snippet is False, omit the "body" key
        if hasattr(self, "show_snippet") and not self.show_snippet:
            filtered_results = [
                {"title": r["title"], "href": r["href"]} for r in filtered_results
            ]

        return filtered_results
    except Exception as e:
        return {"error": str(e)}


def patch_websearch_in_globals(entry_id: str):
    """Monkey-patch WebSearchAPI instances in multi_turn_utils globals().

    After execute_multi_turn_func_call() creates class instances in the
    multi_turn_utils module's globals(), find any WebSearchAPI instances
    matching the entry_id and replace their search_engine_query method.

    Args:
        entry_id: The BFCL entry ID used to identify the correct instances.
    """
    sanitized_id = re.sub(r"[-./:]", "_", entry_id)
    mt_globals = vars(_mt_utils)
    for key, val in list(mt_globals.items()):
        if "WebSearchAPI" in key and sanitized_id in key:
            val.search_engine_query = types.MethodType(_ddg_search_engine_query, val)


def cleanup_globals(entry_id: str):
    """Remove class instances from multi_turn_utils globals().

    Args:
        entry_id: The BFCL entry ID used to identify instances to clean up.
    """
    sanitized_id = re.sub(r"[-./:]", "_", entry_id)
    mt_globals = vars(_mt_utils)
    keys_to_remove = [
        key for key in mt_globals if sanitized_id in key and "_instance" in key
    ]
    for key in keys_to_remove:
        del mt_globals[key]
