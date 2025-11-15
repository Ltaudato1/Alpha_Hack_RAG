CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS text_embeddings (
    id BIGSERIAL PRIMARY KEY,
    text_content TEXT NOT NULL,
    embedding vector(384),
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS embedding_idx 
ON text_embeddings 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO postgres;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO postgres;