import os
import urllib.request
import urllib.parse
import json
import ssl
import re
from typing import Optional, Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from sqlalchemy.orm import Session
from app.models.db_models import Movie

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

TMDB_KEY = "15d2ea6d0dc1d476efbca3eba2b9bbfb"

LINKS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "links.json")
LINKS_MAP = {}
if os.path.exists(LINKS_FILE):
    try:
        with open(LINKS_FILE, "r", encoding="utf-8") as f:
            LINKS_MAP = {int(k): v for k, v in json.load(f).items()}
    except Exception:
        pass

def clean_movie_title(raw_title: str):
    """
    Cleans movie titles for search:
    'Don't Look Back (Ne te retourne pas) (2009)' -> ('Don\'t Look Back', 2009)
    'Chinese Ghost Story II, A (Sien nui yau wan II yan gaan do) (1990)' -> ('A Chinese Ghost Story II', 1990)
    """
    year_match = re.search(r'\((\d{4})\)', raw_title)
    year = int(year_match.group(1)) if year_match else None
    
    # Remove trailing year and extra parentheses
    title = re.sub(r'\s*\(\d{4}\).*', '', raw_title).strip()
    
    # If contains alternative foreign title in parens: "Chinese Ghost Story II, A (Sien nui...)" -> "Chinese Ghost Story II, A"
    if '(' in title:
        title = title.split('(')[0].strip()
        
    # Handle English / French / Spanish trailing articles
    for art in [', The', ', A', ', An', ', La', ', Le', ', Les', ', Il', ', El', ', Die', ', Das', ', Der', ', Un', ', Une']:
        if title.endswith(art):
            title = art.replace(', ', '').strip() + ' ' + title[:-len(art)].strip()
            break
            
    return title.strip(), year

def fetch_tmdb_metadata(tmdb_id: Optional[int] = None, title: Optional[str] = None, year: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """Fetch poster, backdrop, overview, runtime from TMDB."""
    try:
        data = None
        if tmdb_id:
            url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={TMDB_KEY}"
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, context=ctx, timeout=3) as resp:
                data = json.loads(resp.read().decode('utf-8'))
        
        if not data and title:
            clean_t, y = clean_movie_title(title)
            query_str = urllib.parse.quote(clean_t)
            search_url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_KEY}&query={query_str}"
            if y:
                search_url += f"&year={y}"
            req = urllib.request.Request(search_url)
            with urllib.request.urlopen(req, context=ctx, timeout=3) as resp:
                search_res = json.loads(resp.read().decode('utf-8'))
                results = search_res.get('results', [])
                if not results and y:
                    # Retry without year constraint in case release year slightly differs in TMDB
                    search_url_noyear = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_KEY}&query={query_str}"
                    with urllib.request.urlopen(urllib.request.Request(search_url_noyear), context=ctx, timeout=3) as r_resp:
                        results = json.loads(r_resp.read().decode('utf-8')).get('results', [])
                if results:
                    data = results[0]
                    # Also fetch details for runtime
                    try:
                        m_id = data.get('id')
                        detail_url = f"https://api.themoviedb.org/3/movie/{m_id}?api_key={TMDB_KEY}"
                        with urllib.request.urlopen(urllib.request.Request(detail_url), context=ctx, timeout=2) as d_resp:
                            data = json.loads(d_resp.read().decode('utf-8'))
                    except Exception:
                        pass
        
        if not data:
            return None
            
        poster_path = data.get('poster_path')
        backdrop_path = data.get('backdrop_path')
        overview = data.get('overview') or ""
        runtime = data.get('runtime') or 0
        
        poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else ""
        backdrop_url = f"https://image.tmdb.org/t/p/w1280{backdrop_path}" if backdrop_path else (poster_url if poster_url else "")
        
        return {
            "poster_url": poster_url,
            "backdrop_url": backdrop_url,
            "overview": overview,
            "runtime": runtime
        }
    except Exception:
        return None

def ensure_movie_posters(movies: List[Movie], db: Session):
    """
    Guarantees all movies passed have posters populated.
    If any movie lacks a poster, it is fetched on-the-fly via TMDB and saved to DB.
    """
    missing = [m for m in movies if not m.poster_url]
    if not missing:
        return
        
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_map = {
            executor.submit(
                fetch_tmdb_metadata,
                tmdb_id=LINKS_MAP.get(m.movie_id, {}).get("tmdb_id"),
                title=m.title,
                year=m.year,
            ): m
            for m in missing
        }
        updated = False
        for f in as_completed(future_map):
            m = future_map[f]
            try:
                meta = f.result()
                if meta and meta.get("poster_url"):
                    m.poster_url = meta["poster_url"]
                    if meta.get("backdrop_url"):
                        m.backdrop_url = meta["backdrop_url"]
                    if meta.get("overview") and (not m.description or "audience rating count" in m.description):
                        m.description = meta["overview"]
                    if meta.get("runtime") and meta["runtime"] > 0:
                        m.runtime = meta["runtime"]
                    updated = True
            except Exception:
                pass
                
        if updated:
            try:
                db.commit()
            except Exception:
                db.rollback()
