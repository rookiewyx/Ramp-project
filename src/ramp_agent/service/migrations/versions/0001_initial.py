"""初始结构：报销单、审核记录、财务复核、审计日志、发票登记

Revision ID: 0001
Revises: （无，第一版）
Create Date: 2026-11-11

由模型自动生成后人工检查过。引入 Alembic 之前用 create_all 建的库，结构与此一致，
启动时会被直接标记为这个版本（见 migrate.BASELINE）。
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import mysql

revision = '0001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table('audit_events',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('expense_id', sa.String(length=32), nullable=False),
    sa.Column('event', sa.String(length=32), nullable=False),
    sa.Column('actor', sa.String(length=64), nullable=False),
    sa.Column('detail', sa.JSON(), nullable=False),
    sa.Column('at', sa.DateTime().with_variant(mysql.DATETIME(fsp=6), 'mysql'), nullable=False),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_audit_events'))
    )
    with op.batch_alter_table('audit_events', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_audit_events_expense_id'), ['expense_id'], unique=False)

    op.create_table('expenses',
    sa.Column('id', sa.String(length=32), nullable=False),
    sa.Column('idempotency_key', sa.String(length=128), nullable=True),
    sa.Column('request_hash', sa.String(length=64), nullable=False),
    sa.Column('report_id', sa.String(length=64), nullable=False),
    sa.Column('employee_id', sa.String(length=64), nullable=False),
    sa.Column('employee_name', sa.String(length=64), nullable=False),
    sa.Column('category', sa.String(length=32), nullable=False),
    sa.Column('amount', sa.Numeric(precision=12, scale=2), nullable=False),
    sa.Column('submit_date', sa.String(length=10), nullable=False),
    sa.Column('note', sa.Text(), nullable=False),
    sa.Column('city', sa.String(length=64), nullable=True),
    sa.Column('trip_start', sa.String(length=10), nullable=True),
    sa.Column('trip_end', sa.String(length=10), nullable=True),
    sa.Column('nights', sa.Integer(), nullable=True),
    sa.Column('headcount', sa.Integer(), nullable=True),
    sa.Column('preapproval_id', sa.String(length=64), nullable=True),
    sa.Column('file_key', sa.String(length=255), nullable=True),
    sa.Column('file_name', sa.String(length=255), nullable=True),
    sa.Column('file_sha256', sa.String(length=64), nullable=True),
    sa.Column('file_size', sa.Integer(), nullable=True),
    sa.Column('status', sa.String(length=16), nullable=False),
    sa.Column('decision', sa.String(length=16), nullable=True),
    sa.Column('final_decision', sa.String(length=16), nullable=True),
    sa.Column('attempts', sa.Integer(), nullable=False),
    sa.Column('error', sa.Text(), nullable=True),
    sa.Column('created_at', sa.DateTime().with_variant(mysql.DATETIME(fsp=6), 'mysql'), nullable=False),
    sa.Column('updated_at', sa.DateTime().with_variant(mysql.DATETIME(fsp=6), 'mysql'), nullable=False),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_expenses')),
    sa.UniqueConstraint('idempotency_key', name=op.f('uq_expenses_idempotency_key'))
    )
    with op.batch_alter_table('expenses', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_expenses_employee_id'), ['employee_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_expenses_file_sha256'), ['file_sha256'], unique=False)
        batch_op.create_index('ix_expenses_status_decision', ['status', 'decision'], unique=False)

    op.create_table('invoice_claims',
    sa.Column('invoice_no', sa.String(length=32), nullable=False),
    sa.Column('expense_id', sa.String(length=32), nullable=False),
    sa.Column('created_at', sa.DateTime().with_variant(mysql.DATETIME(fsp=6), 'mysql'), nullable=False),
    sa.PrimaryKeyConstraint('invoice_no', name=op.f('pk_invoice_claims'))
    )
    with op.batch_alter_table('invoice_claims', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_invoice_claims_expense_id'), ['expense_id'], unique=False)

    op.create_table('overrides',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('expense_id', sa.String(length=32), nullable=False),
    sa.Column('reviewer', sa.String(length=64), nullable=False),
    sa.Column('decision', sa.String(length=16), nullable=False),
    sa.Column('note', sa.Text(), nullable=False),
    sa.Column('created_at', sa.DateTime().with_variant(mysql.DATETIME(fsp=6), 'mysql'), nullable=False),
    sa.ForeignKeyConstraint(['expense_id'], ['expenses.id'], name=op.f('fk_overrides_expense_id_expenses')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_overrides'))
    )
    with op.batch_alter_table('overrides', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_overrides_expense_id'), ['expense_id'], unique=False)

    op.create_table('review_records',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('expense_id', sa.String(length=32), nullable=False),
    sa.Column('decision', sa.String(length=16), nullable=False),
    sa.Column('results', sa.JSON(), nullable=False),
    sa.Column('explanation', sa.Text(), nullable=False),
    sa.Column('extraction', sa.JSON(), nullable=False),
    sa.Column('llm_calls', sa.Integer(), nullable=False),
    sa.Column('vlm_calls', sa.Integer(), nullable=False),
    sa.Column('tokens', sa.Integer(), nullable=False),
    sa.Column('latency_ms', sa.Integer(), nullable=False),
    sa.Column('versions', sa.JSON(), nullable=False),
    sa.Column('created_at', sa.DateTime().with_variant(mysql.DATETIME(fsp=6), 'mysql'), nullable=False),
    sa.ForeignKeyConstraint(['expense_id'], ['expenses.id'], name=op.f('fk_review_records_expense_id_expenses')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_review_records'))
    )
    with op.batch_alter_table('review_records', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_review_records_expense_id'), ['expense_id'], unique=False)



def downgrade() -> None:
    # 直接删表，索引随表一起删掉。不要先单独删索引：MySQL 不允许删除外键正在使用的索引
    # （错误 1553，在 MySQL 上跑迁移测试时发现）。先删引用别人的表，再删被引用的表。
    for table in ("review_records", "overrides", "invoice_claims", "expenses", "audit_events"):
        op.drop_table(table)
