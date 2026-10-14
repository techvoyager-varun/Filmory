from app.ml.architectures import NCFBaseline, NCFHybrid, SequentialTransformer

# Training only needs architectures; importing them must not require database
# credentials or eagerly initialize the serving stack.
def __getattr__(name):
    if name == "model_service":
        from app.ml.model_service import model_service
        return model_service
    if name in {"get_personalized_recommendations", "get_similar_movies", "get_cold_start_recommendations",
                "get_popular_movies", "get_trending_movies", "get_movies_by_genre"}:
        from app.ml import recommender
        return getattr(recommender, name)
    raise AttributeError(name)
