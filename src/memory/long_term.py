import json
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine


class ScoredDoc(BaseModel):
    document: Document
    score: float

    model_config = {"arbitrary_types_allowed": True}


class LongTermMemory:
    def __init__(self, postgres_url: str, embeddings: Embeddings, table: str = "documents"):
        self._postgres_url = postgres_url
        self._embeddings = embeddings
        self._table = table
        self._engine: Optional[AsyncEngine] = None

    async def _get_engine(self) -> AsyncEngine:
        if self._engine is None:
            self._engine = create_async_engine(self._postgres_url)
        return self._engine

    async def embed_and_store(self, content: str, metadata: dict = {}) -> None:
        engine = await self._get_engine()
        vector = await self._embeddings.aembed_query(content)
        vector_str = "[" + ",".join(str(v) for v in vector) + "]"
        row_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    f"INSERT INTO {self._table} (id, embedding, content, metadata, created_at) "
                    f"VALUES (:id, :embedding::vector, :content, :metadata::jsonb, :created_at)"
                ),
                {
                    "id": row_id,
                    "embedding": vector_str,
                    "content": content,
                    "metadata": json.dumps(metadata),
                    "created_at": now,
                },
            )

    async def retrieve(self, query: str, k: int = 5) -> list[Document]:
        engine = await self._get_engine()
        vector = await self._embeddings.aembed_query(query)
        vector_str = "[" + ",".join(str(v) for v in vector) + "]"
        async with engine.connect() as conn:
            result = await conn.execute(
                text(
                    f"SELECT content, metadata FROM {self._table} "
                    f"ORDER BY embedding <-> :query_vec::vector LIMIT :k"
                ),
                {"query_vec": vector_str, "k": k},
            )
            rows = result.fetchall()
        return [
            Document(page_content=row[0], metadata=json.loads(row[1]) if row[1] else {})
            for row in rows
        ]

    async def grade_relevance(
        self, query: str, docs: list[Document], model: Any
    ) -> list[ScoredDoc]:
        scored: list[ScoredDoc] = []
        for doc in docs:
            prompt = (
                f"Rate the relevance of this document to the query on a scale from 0.0 to 1.0. "
                f"Reply with only a number.\n\nQuery: {query}\n\nDocument: {doc.page_content}"
            )
            response = await model.ainvoke([HumanMessage(content=prompt)])
            try:
                score = float(response.content.strip())
            except ValueError:
                score = 0.0
            scored.append(ScoredDoc(document=doc, score=score))
        return scored

    async def rewrite_query(self, original: str, feedback: str, model: Any) -> str:
        prompt = (
            f"Rewrite this search query to improve retrieval based on the feedback.\n\n"
            f"Original query: {original}\nFeedback: {feedback}\n\nRewritten query:"
        )
        response = await model.ainvoke([HumanMessage(content=prompt)])
        return response.content.strip()

    async def corrective_rag(
        self,
        query: str,
        model: Any,
        k: int = 5,
        relevance_threshold: float = 0.5,
    ) -> list[Document]:
        docs = await self.retrieve(query, k=k)
        scored = await self.grade_relevance(query, docs, model)
        passing = [s.document for s in scored if s.score >= relevance_threshold]
        if not passing:
            rewritten = await self.rewrite_query(
                query, "No relevant documents found — broaden the search.", model
            )
            docs = await self.retrieve(rewritten, k=k)
            return docs
        return passing
