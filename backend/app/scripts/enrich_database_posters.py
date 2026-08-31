import os
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from sqlalchemy.orm import Session
from app.database import SessionLocal
from app.models.db_models import Movie
from app.scripts.tmdb_helper import get_movielens_links, process_movie

def run_enrichment(limit=None):
    print("Starting movie poster enrichment...")
    links_map = get_movielens_links()
    
    db = SessionLocal()
    try:
        # Prioritize movies ordered by popularity / rating count first, so top movies get posters immediately
        query = db.query(Movie).filter(
            (Movie.poster_url == "") | (Movie.poster_url.is_(None))
        ).order_by(Movie.rating_count.desc(), Movie.rating.desc())
        
        if limit:
            query = query.limit(limit)
            
        movies_to_process = query.all()
        total_to_process = len(movies_to_process)
        print(f"Found {total_to_process} movies missing posters.")
        
        movie_items = [
            {"movie_id": m.movie_id, "title": m.title, "year": m.year}
            for m in movies_to_process
        ]
        
        batch_size = 50
        updated_count = 0
        
        print(f"Processing movies using 25 parallel workers...")
        with ThreadPoolExecutor(max_workers=25) as executor:
            future_to_movie = {
                executor.submit(process_movie, item, links_map): item["movie_id"]
                for item in movie_items
            }
            
            pending_updates = {}
            for i, future in enumerate(as_completed(future_to_movie), 1):
                try:
                    m_id, meta = future.result()
                    if meta and meta.get("poster_url"):
                        pending_updates[m_id] = meta
                except Exception as e:
                    pass
                
                # Commit in batches of 50
                if len(pending_updates) >= batch_size or i == total_to_process:
                    if pending_updates:
                        for m_id, meta in pending_updates.items():
                            movie = db.query(Movie).filter(Movie.movie_id == m_id).first()
                            if movie:
                                if meta.get("poster_url"):
                                    movie.poster_url = meta["poster_url"]
                                if meta.get("backdrop_url"):
                                    movie.backdrop_url = meta["backdrop_url"]
                                if meta.get("overview") and (not movie.description or "audience rating count" in movie.description):
                                    movie.description = meta["overview"]
                                if meta.get("runtime") and meta["runtime"] > 0:
                                    movie.runtime = meta["runtime"]
                                updated_count += 1
                        db.commit()
                        print(f"[{i}/{total_to_process}] Saved batch. Total movies with new posters: {updated_count}")
                        pending_updates = {}
                        
        print(f"Done! Successfully updated {updated_count} movies with real TMDB posters & backdrops.")
        
    finally:
        db.close()

if __name__ == "__main__":
    # Enrich all remaining movies in the database
    run_enrichment(limit=None)
