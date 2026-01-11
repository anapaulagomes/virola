import json

import llm
from pydantic import BaseModel

from vir.config import logger

LLM_MODEL = "llama3.2:latest"
PROMPT = """
Give me a short descriptive name for a potential syndrome for a syndromic surveillance system
composed by the following codes:

TERMINOLOGY LABELS (COUNT)

{labels_in_human_format}
"""


class ClusterDescription(BaseModel):
    short_description: str
    description: str
    abbreviation: str


def describe_cluster(terminology_labels_count: dict[str, int]):
    if not terminology_labels_count:
        return {"short_description": "", "description": ""}
    labels_in_human_format = "\n".join(
        [f"{label} ({count})" for label, count in terminology_labels_count.items()]
    )
    try:
        model = llm.get_model(LLM_MODEL)
    except KeyError:
        logger.error("Model not available. Check Ollama installation.")
        return {field: "" for field in ClusterDescription.model_fields}

    response = model.prompt(
        PROMPT.format(labels_in_human_format=labels_in_human_format), schema=ClusterDescription
    )
    return json.loads(response.text())
