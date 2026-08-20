"""Initial database schema

Revision ID: 0001_initial_tables
Revises: 
Create Date: 2026-08-20 20:30:00.000000

"""
from alembic import op
import sqlalchemy as sa

revision = '0001_initial_tables'
down_revision = None
branch_labels = None
depends_on = None

def upgrade() -> None:
    # Users
    op.create_table(
        'users',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True, index=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('email', sa.String(255), unique=True, index=True, nullable=False),
        sa.Column('password_hash', sa.String(255), nullable=False),
        sa.Column('model_user_id', sa.Integer(), nullable=True, index=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )

    # Movies
    op.create_table(
        'movies',
        sa.Column('movie_id', sa.BigInteger(), primary_key=True, index=True),
        sa.Column('title', sa.String(512), nullable=False, index=True),
        sa.Column('genres', sa.JSON(), nullable=False),
        sa.Column('year', sa.Integer(), nullable=True, index=True),
        sa.Column('rating', sa.Float(), server_default='0.0', index=True),
        sa.Column('rating_count', sa.Integer(), server_default='0', index=True),
        sa.Column('runtime', sa.Integer(), server_default='0'),
        sa.Column('description', sa.Text(), server_default=''),
        sa.Column('poster_url', sa.Text(), server_default=''),
        sa.Column('backdrop_url', sa.Text(), server_default=''),
    )

    # Interactions
    op.create_table(
        'interactions',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True, index=True),
        sa.Column('user_id', sa.Integer(), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('movie_id', sa.BigInteger(), sa.ForeignKey('movies.movie_id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('type', sa.String(50), nullable=False),
        sa.Column('timestamp', sa.DateTime(), nullable=False, server_default=sa.func.now(), index=True),
    )
    op.create_index('ix_interactions_user_movie_time', 'interactions', ['user_id', 'movie_id', 'timestamp'])

    # Watch History
    op.create_table(
        'watch_history',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True, index=True),
        sa.Column('user_id', sa.Integer(), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('movie_id', sa.BigInteger(), sa.ForeignKey('movies.movie_id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('watched_at', sa.DateTime(), nullable=False, server_default=sa.func.now(), index=True),
        sa.Column('progress', sa.Integer(), server_default='100'),
    )
    op.create_index('ix_watch_history_user_movie', 'watch_history', ['user_id', 'movie_id'])

    # My List
    op.create_table(
        'my_list',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True, index=True),
        sa.Column('user_id', sa.Integer(), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('movie_id', sa.BigInteger(), sa.ForeignKey('movies.movie_id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint('user_id', 'movie_id', name='uq_my_list_user_movie'),
    )

    # Likes
    op.create_table(
        'likes',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True, index=True),
        sa.Column('user_id', sa.Integer(), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('movie_id', sa.BigInteger(), sa.ForeignKey('movies.movie_id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint('user_id', 'movie_id', name='uq_likes_user_movie'),
    )

    # User Genre Preferences
    op.create_table(
        'user_genre_preferences',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True, index=True),
        sa.Column('user_id', sa.Integer(), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('genre', sa.String(100), nullable=False),
        sa.Column('score', sa.Float(), server_default='0.0', nullable=False),
        sa.UniqueConstraint('user_id', 'genre', name='uq_user_genre'),
    )

    # User Preferences
    op.create_table(
        'user_preferences',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True, index=True),
        sa.Column('user_id', sa.Integer(), sa.ForeignKey('users.id', ondelete='CASCADE'), unique=True, nullable=False),
        sa.Column('favorite_genres', sa.JSON(), nullable=False),
        sa.Column('favorite_movie_ids', sa.JSON(), nullable=False),
        sa.Column('onboarding_completed', sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )

def downgrade() -> None:
    op.drop_table('user_preferences')
    op.drop_table('user_genre_preferences')
    op.drop_table('likes')
    op.drop_table('my_list')
    op.drop_table('watch_history')
    op.drop_table('interactions')
    op.drop_table('movies')
    op.drop_table('users')
