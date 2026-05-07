"""Wikipedia client package.

Exposes :func:`search_wikipedia`, the single tool function shared by the
Haiku agent under test (subagent C) and the Opus data-gen workers
(subagent B). One implementation, one cache, no re-implementation.
"""

from wiki.client import search_wikipedia  # re-export

__all__ = ["search_wikipedia"]
