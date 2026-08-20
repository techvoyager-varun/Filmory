from app.ml.architectures import NCFBaseline, NCFHybrid, SequentialTransformer
from app.ml.model_service import model_service
from app.ml.recommender import (
    get_personalized_recommendations,
    get_similar_movies,
    get_cold_start_recommendations,
    get_popular_movies,
    get_trending_movies,
    get_movies_by_genre,
)
