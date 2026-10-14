"""Add opt-in local recommendation study snapshots.

Revision ID: 0003_recommendation_study
Revises: 0002_assistant_tables
"""
from alembic import op
import sqlalchemy as sa

revision = '0003_recommendation_study'
down_revision = '0002_assistant_tables'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table('recommendation_studies',
        sa.Column('id',sa.String(36),primary_key=True),
        sa.Column('user_id',sa.Integer(),sa.ForeignKey('users.id',ondelete='CASCADE'),nullable=False,index=True),
        sa.Column('movies',sa.JSON(),nullable=False),
        sa.Column('policy_snapshot',sa.JSON(),nullable=False),
        sa.Column('answers',sa.JSON(),nullable=False),
        sa.Column('created_at',sa.DateTime(),nullable=False),
        sa.Column('updated_at',sa.DateTime(),nullable=False))


def downgrade():
    op.drop_table('recommendation_studies')
