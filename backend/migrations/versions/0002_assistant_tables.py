"""Add the assistant tables missing from the initial migration.

Revision ID: 0002_assistant_tables
Revises: 0001_initial_tables
"""
from alembic import op
import sqlalchemy as sa

revision = '0002_assistant_tables'
down_revision = '0001_initial_tables'
branch_labels = None
depends_on = None

def upgrade():
    op.create_table('assistant_sessions',
        sa.Column('id',sa.String(36),primary_key=True),
        sa.Column('user_id',sa.Integer(),sa.ForeignKey('users.id',ondelete='CASCADE'),nullable=False,index=True),
        sa.Column('active_intent',sa.JSON(),nullable=False),
        sa.Column('created_at',sa.DateTime(),nullable=False),
        sa.Column('updated_at',sa.DateTime(),nullable=False))
    op.create_table('assistant_messages',
        sa.Column('id',sa.Integer(),primary_key=True,autoincrement=True,index=True),
        sa.Column('session_id',sa.String(36),sa.ForeignKey('assistant_sessions.id',ondelete='CASCADE'),nullable=False,index=True),
        sa.Column('role',sa.String(20),nullable=False),sa.Column('content',sa.Text(),nullable=False),
        sa.Column('intent_snapshot',sa.JSON()),sa.Column('movie_ids',sa.JSON()),
        sa.Column('created_at',sa.DateTime(),nullable=False))
    op.create_table('assistant_feedback',
        sa.Column('id',sa.Integer(),primary_key=True,autoincrement=True,index=True),
        sa.Column('session_id',sa.String(36),sa.ForeignKey('assistant_sessions.id',ondelete='CASCADE'),nullable=False,index=True),
        sa.Column('message_id',sa.Integer()),sa.Column('movie_id',sa.BigInteger()),
        sa.Column('feedback_type',sa.String(50),nullable=False),sa.Column('comment',sa.Text()),
        sa.Column('created_at',sa.DateTime(),nullable=False))
    op.create_table('movie_text_embeddings',
        sa.Column('movie_id',sa.BigInteger(),sa.ForeignKey('movies.movie_id',ondelete='CASCADE'),primary_key=True),
        sa.Column('embedding_model',sa.String(100),nullable=False),
        sa.Column('embedding_dimension',sa.Integer(),nullable=False),
        sa.Column('document_hash',sa.String(64),nullable=False,index=True),
        sa.Column('indexed_at',sa.DateTime(),nullable=False))

def downgrade():
    for table in ['movie_text_embeddings','assistant_feedback','assistant_messages','assistant_sessions']:
        op.drop_table(table)
