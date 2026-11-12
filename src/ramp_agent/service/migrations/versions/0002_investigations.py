"""调查 Agent：调查记录表 investigations，报销单加 agent_status

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-12

只加表和可空字段，不改已有数据：升级时旧的报销单 agent_status 为空，表示没调查过。
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import mysql

revision = '0002'
down_revision = '0001'
branch_labels = None
depends_on = None

TS = sa.DateTime().with_variant(mysql.DATETIME(fsp=6), 'mysql')


def upgrade() -> None:
    with op.batch_alter_table('expenses', schema=None) as batch_op:
        batch_op.add_column(sa.Column('agent_status', sa.String(length=24), nullable=True))

    op.create_table('investigations',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('expense_id', sa.String(length=32), nullable=False),
    sa.Column('status', sa.String(length=24), nullable=False),
    sa.Column('decision_before', sa.String(length=16), nullable=False),
    sa.Column('decision_after', sa.String(length=16), nullable=True),
    sa.Column('question', sa.Text(), nullable=True),
    sa.Column('reply', sa.Text(), nullable=True),
    sa.Column('asked_at', TS, nullable=True),
    sa.Column('replied_at', TS, nullable=True),
    sa.Column('result', sa.JSON(), nullable=False),
    sa.Column('state', sa.JSON(), nullable=True),
    sa.Column('llm_calls', sa.Integer(), nullable=False),
    sa.Column('tool_calls', sa.Integer(), nullable=False),
    sa.Column('tokens', sa.Integer(), nullable=False),
    sa.Column('latency_ms', sa.Integer(), nullable=False),
    sa.Column('created_at', TS, nullable=False),
    sa.Column('updated_at', TS, nullable=False),
    sa.ForeignKeyConstraint(['expense_id'], ['expenses.id'], name=op.f('fk_investigations_expense_id_expenses')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_investigations'))
    )
    with op.batch_alter_table('investigations', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_investigations_expense_id'), ['expense_id'], unique=False)


def downgrade() -> None:
    op.drop_table('investigations')          # 索引随表删除（同 0001 的说明，避免 MySQL 1553）
    with op.batch_alter_table('expenses', schema=None) as batch_op:
        batch_op.drop_column('agent_status')
