import json

import llm
from diskcache import Cache
from pydantic import BaseModel

from vir.config import logger

LLM_MODEL = "llama3.2:1b"
PROMPT = """
Give me a short descriptive name for a potential syndrome for a syndromic surveillance system
composed by the following diagnoses:

DIAGNOSIS (CENTRALITY SCORE — higher means more representative of the cluster)

{labels_in_human_format}
"""
descriptions_cache = Cache("clusters-description")


class ClusterDescription(BaseModel):
    short_description: str
    description: str
    abbreviation: str


def _cache_description_key(labels: list[str]):
    return "-".join([label.replace(" ", "_") for label in labels])


def get_llm_model():
    """Resolve and return the LLM model. Raises KeyError if not available."""
    try:
        return llm.get_model(LLM_MODEL)
    except KeyError:
        logger.error(f"Model '{LLM_MODEL}' not available. Check Ollama installation.")
        raise


def describe_cluster(terminology_labels_count: dict[str, float | int], model=None):
    if not terminology_labels_count:
        return {"short_description": "", "description": "", "abbreviation": ""}
    sorted_dict = dict(
        sorted(terminology_labels_count.items(), key=lambda item: item[1], reverse=True)
    )
    cache_key = _cache_description_key(sorted_dict.keys())

    if descriptions_cache.get(cache_key):
        return descriptions_cache[cache_key]

    labels_in_human_format = "\n".join(
        [f"{label} ({count})" for label, count in sorted_dict.items()]
    )
    if model is None:
        model = get_llm_model()

    response = model.prompt(
        PROMPT.format(labels_in_human_format=labels_in_human_format), schema=ClusterDescription
    )
    cluster_description = json.loads(response.text())
    descriptions_cache[cache_key] = cluster_description
    return cluster_description
