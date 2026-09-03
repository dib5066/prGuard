"""initial schema

Revision ID: e49f85348ad5
Revises: 
Create Date: 2026-08-30 11:20:08.442712

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e49f85348ad5'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create all PRGuard tables."""
    
    # Users table
    op.create_table(
        'users',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('github_username', sa.String(255), unique=True, nullable=False),
        sa.Column('email', sa.String(255), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    
    # GitHub Installations table
    op.create_table(
        'github_installations',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('installation_id', sa.BigInteger(), unique=True, nullable=False),
        sa.Column('account_name', sa.String(255), nullable=False),
        sa.Column('account_type', sa.String(50), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    
    # Repositories table
    op.create_table(
        'repositories',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('github_id', sa.BigInteger(), unique=True, nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('full_name', sa.String(255), unique=True, nullable=False),
        sa.Column('installation_id', sa.BigInteger(), sa.ForeignKey('github_installations.installation_id', ondelete='CASCADE'), nullable=False),
        sa.Column('is_indexed', sa.Boolean(), nullable=False, server_default=sa.text('false')),
        sa.Column('last_indexed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    
    # Pull Requests table
    op.create_table(
        'pull_requests',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('repository_id', sa.Integer(), sa.ForeignKey('repositories.id', ondelete='CASCADE'), nullable=False),
        sa.Column('number', sa.Integer(), nullable=False),
        sa.Column('title', sa.String(255), nullable=False),
        sa.Column('state', sa.String(50), nullable=False),
        sa.Column('base_sha', sa.String(40), nullable=False),
        sa.Column('head_sha', sa.String(40), nullable=False),
        sa.Column('user_login', sa.String(255), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint('repository_id', 'number', name='uq_repository_pull_request'),
    )
    
    # Reviews table
    op.create_table(
        'reviews',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('pull_request_id', sa.Integer(), sa.ForeignKey('pull_requests.id', ondelete='CASCADE'), nullable=False),
        sa.Column('status', sa.String(50), nullable=False, server_default='PENDING'),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
    )
    
    # Findings table
    op.create_table(
        'findings',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('review_id', sa.Integer(), sa.ForeignKey('reviews.id', ondelete='CASCADE'), nullable=False),
        sa.Column('severity', sa.String(50), nullable=False),
        sa.Column('category', sa.String(100), nullable=False),
        sa.Column('title', sa.String(255), nullable=False),
        sa.Column('description', sa.Text(), nullable=False),
        sa.Column('file_path', sa.String(512), nullable=False),
        sa.Column('line_number', sa.Integer(), nullable=True),
        sa.Column('evidence', sa.Text(), nullable=True),
        sa.Column('confidence', sa.Numeric(3, 2), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    
    # Review Runs table
    op.create_table(
        'review_runs',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('review_id', sa.Integer(), sa.ForeignKey('reviews.id', ondelete='CASCADE'), nullable=False),
        sa.Column('agent_name', sa.String(100), nullable=False),
        sa.Column('latency_ms', sa.Integer(), nullable=True),
        sa.Column('tokens_used', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )


def downgrade() -> None:
    """Drop all PRGuard tables in reverse order."""
    op.drop_table('review_runs')
    op.drop_table('findings')
    op.drop_table('reviews')
    op.drop_table('pull_requests')
    op.drop_table('repositories')
    op.drop_table('github_installations')
    op.drop_table('users')
