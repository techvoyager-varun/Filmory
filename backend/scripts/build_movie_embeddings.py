"""
Script to pre-compute and index movie text embeddings using Google Gemini's
text-embedding-004 model.

Usage:
    python -m scripts.build_movie_embeddings [--batch-size 50] [--limit 1000]

Requires:
    - GEMINI_API_KEY set in .env
    - PostgreSQL with pgvector extension enabled (optional; will report status)
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import logging
import sys
import time
from typing import List

from sqlalchemy import text
from app.config import settings
from app.database import engine, get_db
from app.models.db_models import Movie, MovieTextEmbedding

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("filmory.embeddings")


def check_pgvector_installed() -> bool:
    """Check if pgvector extension is available in PostgreSQL."""
    try:
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT extname FROM pg_extension WHERE extname = 'vector'")
            ).fetchone()
            return result is not None
    except Exception as e:
        logger.warning(f"Could not check pgvector extension: {e}")
        return False


def build_embeddings(batch_size: int = 50, limit: int | None = None, force: bool = False):
    """
    Incremental embedding generator with SHA-256 document hashing:
    - Skips movies where document_hash and embedding_model match existing records.
    - Re-indexes when description or model changes.
    """
    if not settings.GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY is not set. Please add it to .env.")
        sys.exit(1)

    try:
        from google import genai
        client = genai.Client(api_key=settings.GEMINI_API_KEY)
    except Exception as e:
        logger.error(f"Failed to initialize Gemini client: {e}")
        sys.exit(1)

    has_pgvector = check_pgvector_installed()
    logger.info(f"pgvector extension installed: {has_pgvector}")
    logger.info(f"Using embedding model: {settings.GEMINI_EMBEDDING_MODEL}")

    # Fetch candidate movies with descriptions
    db = next(get_db())
    try:
        # Load existing hashes to avoid redundant API calls
        existing_records = {
            rec.movie_id: (rec.document_hash, rec.embedding_model)
            for rec in db.query(MovieTextEmbedding).all()
        }

        query = db.query(Movie).filter(Movie.description.isnot(None), Movie.description != "")
        if limit:
            query = query.limit(limit)
        movies = query.all()
        total_movies = len(movies)
        logger.info(f"Found {total_movies} total candidate movies with descriptions.")

        # Filter movies that actually need indexing
        movies_to_index: list[tuple[Movie, str, str]] = []
        for m in movies:
            doc_text = f"Title: {m.title}\nGenres: {', '.join(m.genres or [])}\nSynopsis: {m.description[:400]}"
            doc_hash = hashlib.sha256(doc_text.encode("utf-8")).hexdigest()

            if not force and m.movie_id in existing_records:
                prev_hash, prev_model = existing_records[m.movie_id]
                if prev_hash == doc_hash and prev_model == settings.GEMINI_EMBEDDING_MODEL:
                    continue  # Already up-to-date!

            movies_to_index.append((m, doc_text, doc_hash))

        num_to_index = len(movies_to_index)
        logger.info(f"Movies requiring indexing or re-indexing: {num_to_index}/{total_movies}")

        if num_to_index == 0:
            logger.info("All movie embeddings are up-to-date.")
            return

        processed = 0
        for i in range(0, num_to_index, batch_size):
            batch = movies_to_index[i : i + batch_size]
            texts = [item[1] for item in batch]

            try:
                response = client.models.embed_content(
                    model=settings.GEMINI_EMBEDDING_MODEL,
                    contents=texts,
                )
                embeddings = response.embeddings

                for (m, _, doc_hash), emb in zip(batch, embeddings):
                    dim = len(emb.values) if hasattr(emb, "values") else 768
                    record = db.query(MovieTextEmbedding).filter(MovieTextEmbedding.movie_id == m.movie_id).first()
                    if not record:
                        record = MovieTextEmbedding(
                            movie_id=m.movie_id,
                            embedding_model=settings.GEMINI_EMBEDDING_MODEL,
                            embedding_dimension=dim,
                            document_hash=doc_hash,
                            indexed_at=datetime.datetime.utcnow(),
                        )
                        db.add(record)
                    else:
                        record.embedding_model = settings.GEMINI_EMBEDDING_MODEL
                        record.embedding_dimension = dim
                        record.document_hash = doc_hash
                        record.indexed_at = datetime.datetime.utcnow()

                db.commit()
                processed += len(batch)
                logger.info(f"Embedded batch {i // batch_size + 1}: {processed}/{num_to_index} movies.")
            except Exception as e:
                logger.error(f"Error embedding batch starting at index {i}: {e}")
                db.rollback()
                time.sleep(2)
                continue

            time.sleep(0.5)

        logger.info(f"Embedding pipeline finished. Total newly indexed: {processed}/{num_to_index}")
    finally:
        db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate movie synopsis embeddings with Gemini.")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size for embedding generation")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of movies to embed")
    parser.add_argument("--force", action="store_true", help="Force re-indexing all movies")
    args = parser.parse_args()

    build_embeddings(batch_size=args.batch_size, limit=args.limit, force=args.force)

