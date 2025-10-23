import json

import llm
from pydantic import BaseModel

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
    labels_in_human_format = "\n".join(
        [f"{label} ({count})" for label, count in terminology_labels_count.items()]
    )
    model = llm.get_model(LLM_MODEL)
    response = model.prompt(
        PROMPT.format(labels_in_human_format=labels_in_human_format), schema=ClusterDescription
    )
    return json.loads(response.text())
