import urllib.request
import urllib.parse
import json
import ssl
import io
import zipfile
import csv
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from sqlalchemy.orm import Session
from app.database import SessionLocal
from app.models.db_models import Movie

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

TMDB_KEY = "15d2ea6d0dc1d476efbca3eba2b9bbfb"

def get_movielens_links():
    """Download ml-latest-small links.csv directly."""
    print("Downloading ml-latest-small links...")
    url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    links = {}
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=20) as resp:
            z = zipfile.ZipFile(io.BytesIO(resp.read()))
            with z.open("ml-latest-small/links.csv") as f:
                reader = csv.DictReader(io.TextIOWrapper(f))
                for row in reader:
                    m_id = int(row['movieId'])
                    tmdb_id = row.get('tmdbId')
                    imdb_id = row.get('imdbId')
                    links[m_id] = {
                        'tmdb_id': int(tmdb_id) if tmdb_id and tmdb_id.strip() else None,
                        'imdb_id': imdb_id.strip() if imdb_id and imdb_id.strip() else None
                    }
        print(f"Loaded {len(links)} links from MovieLens dataset.")
    except Exception as e:
        print(f"Warning: Could not download links.csv: {e}")
    return links

def clean_movie_title(raw_title: str):
    """
    Extracts clean title and year from MovieLens title like:
    'American President, The (1995)' -> ('The American President', 1995)
    'City of Lost Children, The (Cité des enfants perdus, La) (1995)' -> ('The City of Lost Children', 1995)
    """
    # Extract year if present
    year_match = re.search(r'\((\d{4})\)', raw_title)
    year = int(year_match.group(1)) if year_match else None
    
    # Remove year part
    title = re.sub(r'\s*\(\d{4}\).*', '', raw_title).strip()
    
    # Handle alternative foreign titles in parens like: "City of Lost Children, The (Cité des...)"
    if '(' in title:
        title = title.split('(')[0].strip()
        
    # Handle trailing articles: "President, The" -> "The President"
    for art in [', The', ', A', ', An', ', La', ', Le', ', Les', ', Il', ', El', ', Die', ', Das', ', Der']:
        if title.endswith(art):
            title = art.replace(', ', '').strip() + ' ' + title[:-len(art)].strip()
            break
            
    return title.strip(), year

def fetch_tmdb_metadata(tmdb_id=None, title=None, year=None):
    """Fetch poster, backdrop, overview, runtime from TMDB."""
    try:
        data = None
        if tmdb_id:
            url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={TMDB_KEY}"
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, context=ctx, timeout=4) as resp:
                data = json.loads(resp.read().decode('utf-8'))
        
        if not data and title:
            clean_t, y = clean_movie_title(title)
            query_str = urllib.parse.quote(clean_t)
            search_url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_KEY}&query={query_str}"
            if y:
                search_url += f"&year={y}"
            req = urllib.request.Request(search_url)
            with urllib.request.urlopen(req, context=ctx, timeout=4) as resp:
                search_res = json.loads(resp.read().decode('utf-8'))
                results = search_res.get('results', [])
                if results:
                    data = results[0]
                    # Also fetch details for runtime
                    try:
                        m_id = data.get('id')
                        detail_url = f"https://api.themoviedb.org/3/movie/{m_id}?api_key={TMDB_KEY}"
                        with urllib.request.urlopen(urllib.request.Request(detail_url), context=ctx, timeout=3) as d_resp:
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

def process_movie(movie_item, links_map):
    m_id = movie_item["movie_id"]
    title = movie_item["title"]
    
    link_info = links_map.get(m_id, {})
    tmdb_id = link_info.get('tmdb_id')
    
    meta = fetch_tmdb_metadata(tmdb_id=tmdb_id, title=title, year=movie_item.get("year"))
    return m_id, meta
