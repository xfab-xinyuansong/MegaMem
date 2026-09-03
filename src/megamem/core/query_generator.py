from omegaconf import DictConfig
from pydantic import BaseModel, Field
from megamem.utils.llm import ChatCompletionModel

PROMPT_GENERATE_QUERIES = """
ROLE
You are the Memory Query Generator. Your job is to transform a context into concise, specific queries for retrieving relevant memory. The context may not explicitly state a task — you must infer what task is implied and what information is required to support it.

INSTRUCTIONS
1. Carefully read the full context.
2. Infer the most likely task or intent implied by the context (e.g., troubleshoot, summarize, explain, retrieve past actions).
3. Decide what information from memory would be necessary to accomplish that task.
4. Generate one or multiple **concise (6–14 words)** queries capturing exactly that need.
5. There are multiple types of memories available such as "factual","procedural" and "episodic" memory.

Here is the context:
{context}

"""

PROMPT_EXTRACT_KEYWORDS = """
ROLE
You are a Keyword Extractor for memory retrieval. Your job is to identify the most important keywords and phrases from a context that would be useful for keyword-based search.

INSTRUCTIONS
1. Carefully read the context.
2. Extract meaningful short phrases **(2-4 words)** that can be used to perform exact-match searches to retrieve relevant information.
3. Keep phrases concise, but avoid single-word names unless absolutely necessary. The phrases do not need to include the user name. Avoid using "and" in the phrases.
4. Return 1-4 keywords/phrases depending on context complexity.

Here is the context:
{context}

"""


class MemoryQuery(BaseModel):
    query: str = Field(
        description="A short, precise query capturing exactly that needed for retrieval."
    )


class MemoryQueries(BaseModel):
    queries: list[MemoryQuery] = Field(description="a list of queries")


class Keyword(BaseModel):
    keyword: str = Field(
        description="A keyword, entity, or short phrase useful for search."
    )


class Keywords(BaseModel):
    keywords: list[Keyword] = Field(description="a list of keywords and phrases")


class QueryGenerator:

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self._model_client = ChatCompletionModel(cfg)  # LLM client for prompt invocations
        pass

    def generate_queries(
        self,
        context: str,
    ) -> list[str]:

        # Hand the context to the LLM and parse its structured response.
        prompt_args = {
            "context": context,
        }
        results: MemoryQueries = self._model_client.invoke(
            input=PROMPT_GENERATE_QUERIES,
            prompt_args=prompt_args,
            response_format=MemoryQueries,
        )

        return [item.query for item in results.queries]

    def extract_keywords(
        self,
        context: str,
    ) -> list[str]:
        """
        Pull keywords / short phrases from a context for keyword-style retrieval.

        Args:
            context: The context to extract keywords from

        Returns:
            List of keywords and phrases
        """
        prompt_args = {
            "context": context,
        }
        results: Keywords = self._model_client.invoke(
            input=PROMPT_EXTRACT_KEYWORDS,
            prompt_args=prompt_args,
            response_format=Keywords,
        )

        return [item.keyword for item in results.keywords]
