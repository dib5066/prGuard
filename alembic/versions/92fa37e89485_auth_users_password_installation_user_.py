"""auth: users password + installation user_id + review repository_id

Revision ID: 92fa37e89485
Revises: 8b949b5fa9e1
Create Date: 2026-09-03 21:26:48.884091

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '92fa37e89485'
down_revision: Union[str, Sequence[str], None] = '8b949b5fa9e1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# NOTE: the LangGraph checkpoint* tables live in this database but are owned
# by AsyncPostgresSaver, not Alembic — they are deliberately not touched here.


def upgrade() -> None:
    # --- github_installations: link to a user + lifecycle flags ------------
    op.add_column('github_installations', sa.Column('account_id', sa.BigInteger(), nullable=True))
    op.add_column('github_installations', sa.Column('user_id', sa.Integer(), nullable=True))
    op.add_column('github_installations', sa.Column('suspended_at', sa.DateTime(timezone=True), nullable=True))
    op.add_column('github_installations', sa.Column('deleted_at', sa.DateTime(timezone=True), nullable=True))
    op.create_foreign_key(
        'fk_github_installations_user_id', 'github_installations', 'users',
        ['user_id'], ['id'], ondelete='SET NULL',
    )

    # --- reviews: denormalized repository_id (backfilled) -----------------
    op.add_column('reviews', sa.Column('repository_id', sa.Integer(), nullable=True))
    op.create_foreign_key(
        'fk_reviews_repository_id', 'reviews', 'repositories',
        ['repository_id'], ['id'], ondelete='CASCADE',
    )
    op.execute(
        """
        UPDATE reviews r
        SET repository_id = pr.repository_id
        FROM pull_requests pr
        WHERE r.pull_request_id = pr.id
          AND r.repository_id IS NULL
        """
    )

    # --- users: real credential columns ---------------------------------
    op.add_column(
        'users',
        sa.Column('password_hash', sa.String(length=255),
                  nullable=False, server_default=''),
    )
    op.add_column(
        'users',
        sa.Column('updated_at', sa.DateTime(timezone=True),
                  nullable=False, server_default=sa.func.now()),
    )
    op.alter_column('users', 'email', existing_type=sa.VARCHAR(length=255), nullable=False)
    op.alter_column('users', 'github_username', existing_type=sa.VARCHAR(length=255), nullable=True)
    op.drop_constraint(op.f('users_github_username_key'), 'users', type_='unique')
    op.create_unique_constraint('uq_users_email', 'users', ['email'])

    # server_defaults were only needed to fill any existing rows.
    op.alter_column('users', 'password_hash', server_default=None)


def downgrade() -> None:
    op.drop_constraint('uq_users_email', 'users', type_='unique')
    op.create_unique_constraint(op.f('users_github_username_key'), 'users', ['github_username'])
    op.alter_column('users', 'github_username', existing_type=sa.VARCHAR(length=255), nullable=False)
    op.alter_column('users', 'email', existing_type=sa.VARCHAR(length=255), nullable=True)
    op.drop_column('users', 'updated_at')
    op.drop_column('users', 'password_hash')

    op.drop_constraint('fk_reviews_repository_id', 'reviews', type_='foreignkey')
    op.drop_column('reviews', 'repository_id')

    op.drop_constraint('fk_github_installations_user_id', 'github_installations', type_='foreignkey')
    op.drop_column('github_installations', 'deleted_at')
    op.drop_column('github_installations', 'suspended_at')
    op.drop_column('github_installations', 'user_id')
    op.drop_column('github_installations', 'account_id')
