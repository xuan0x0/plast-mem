# Graph Memory (PostgreSQL + Recursive CTE) Integration Plan

This document proposes a **PostgreSQL-native** “graph memory” layer for Plast Mem that:

1) models a graph using **multiple relational tables**, and
2) performs **k-hop traversal with recursive CTEs**,

while aiming for **Mnemis/Graphiti-style SOTA long-term memory retrieval** by adding a *deliberate* graph route on top of the existing fast hybrid retrieval.

---

## Why Graph, given Plast Mem already has episodic + semantic?

Current retrieval is strong for “direct cue → direct match”:

- `semantic_memory`: BM25 (ParadeDB) + vector RRF (no decay)
- `episodic_memory`: BM25 + vector RRF × FSRS retrievability (decayed)

Where scores often drop (LoCoMo / LongMemEval multi-session & multi-hop style questions) is:

- **indirect cues** (“what about their teammate?” when the name never appears in query)
- **bridging** across sessions (A mentions B in episode X; later a query mentions B and needs A)
- **top-down recall** (topic/community first, then relevant entities/episodes)

Graph memory adds:

- entity-centric multi-hop expansion
- topic/community “global selection” (hierarchical route)
- evidence traceability (edges tied back to facts/episodes)

---

## Design: Dual-route retrieval (fast + deliberate)

### Route 1: Existing fast hybrid retrieval (keep as-is)

- `SemanticMemory::retrieve*`
- `EpisodicMemory::retrieve`

### Route 2: Graph route (new)

1) Seed entities/communities from query (BM25 + vector)
2) Expand via recursive CTE (k hops) with attenuation
3) Convert reached entities → candidate facts/episodes via join tables
4) Fuse with Route 1 using RRF (and FSRS for episodic final ranking)

This mirrors “dual-route” ideas without replacing the current pipeline.

---

## Minimal schema (multiple tables)

All tables are **conversation-scoped** (keep isolation consistent with current design).

### Conversation scope note

In Plast Mem, `conversation_id` is effectively a **stable (user, assistant) pair** memory space. If the same user talks to a different assistant, that should be a different `conversation_id`. This makes `conversation_id` the natural shard boundary for graph construction, community clustering, and recursive traversal.

### 1) `graph_entity`

Canonical entity nodes (people, orgs, projects, tools, places, etc.)

```sql
CREATE TABLE graph_entity (
  id              uuid PRIMARY KEY,
  conversation_id uuid NOT NULL,
  canonical       text NOT NULL,
  normalized      text NOT NULL,
  embedding       vector(1024) NOT NULL,
  created_at      timestamptz NOT NULL DEFAULT now(),
  -- optional bookkeeping
  updated_at      timestamptz NOT NULL DEFAULT now(),
  UNIQUE (conversation_id, normalized)
);

ALTER TABLE graph_entity
  ADD COLUMN IF NOT EXISTS search_text text
  GENERATED ALWAYS AS (canonical) STORED;

CREATE INDEX IF NOT EXISTS idx_graph_entity_embedding_hnsw
  ON graph_entity USING hnsw (embedding vector_ip_ops);

CREATE INDEX IF NOT EXISTS idx_graph_entity_bm25
  ON graph_entity USING bm25 (id, (search_text::pdb.icu), created_at)
  WITH (key_field='id');
```

**Normalization** (done in Rust):

- lowercase
- trim punctuation
- collapse whitespace
- optionally strip honorifics (“mr”, “dr”) and role tokens (“user”, “assistant”)

### 2) `graph_edge`

Entity-to-entity edges for association + relation-like links (co-occur, same fact, same episode, alias).

```sql
CREATE TABLE graph_edge (
  id              uuid PRIMARY KEY,
  conversation_id uuid NOT NULL,
  src_entity_id   uuid NOT NULL REFERENCES graph_entity(id) ON DELETE CASCADE,
  dst_entity_id   uuid NOT NULL REFERENCES graph_entity(id) ON DELETE CASCADE,
  edge_type       text NOT NULL,
  weight          real NOT NULL,
  evidence        jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at      timestamptz NOT NULL DEFAULT now(),
  invalid_at      timestamptz NULL,
  UNIQUE (conversation_id, src_entity_id, dst_entity_id, edge_type)
);

CREATE INDEX IF NOT EXISTS idx_graph_edge_src
  ON graph_edge (conversation_id, src_entity_id, edge_type)
  WHERE invalid_at IS NULL;

CREATE INDEX IF NOT EXISTS idx_graph_edge_dst
  ON graph_edge (conversation_id, dst_entity_id, edge_type)
  WHERE invalid_at IS NULL;
```

Recommended `edge_type` values:

- `cooccur_fact` (entities appear together in one semantic fact)
- `cooccur_episode` (entities appear together in one episode)
- `alias` (normalized alias → canonical merge)

### 3) `graph_entity_fact` (entity ↔ semantic facts)

```sql
CREATE TABLE graph_entity_fact (
  conversation_id uuid NOT NULL,
  entity_id       uuid NOT NULL REFERENCES graph_entity(id) ON DELETE CASCADE,
  semantic_id     uuid NOT NULL REFERENCES semantic_memory(id) ON DELETE CASCADE,
  weight          real NOT NULL DEFAULT 1.0,
  created_at      timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (entity_id, semantic_id)
);

CREATE INDEX IF NOT EXISTS idx_graph_entity_fact_semantic
  ON graph_entity_fact (conversation_id, semantic_id);
```

### 4) `graph_entity_episode` (entity ↔ episodic episodes)

```sql
CREATE TABLE graph_entity_episode (
  conversation_id uuid NOT NULL,
  entity_id       uuid NOT NULL REFERENCES graph_entity(id) ON DELETE CASCADE,
  episodic_id     uuid NOT NULL REFERENCES episodic_memory(id) ON DELETE CASCADE,
  weight          real NOT NULL DEFAULT 1.0,
  created_at      timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (entity_id, episodic_id)
);

CREATE INDEX IF NOT EXISTS idx_graph_entity_episode_episodic
  ON graph_entity_episode (conversation_id, episodic_id);
```

### 5) Optional: `graph_community` + membership (hierarchical route)

If you want a Mnemis-like top-down “topic traversal” with minimal complexity:

- cluster entities into topic buckets (by **embedding similarity**, optionally informed by co-occur edges)
- summarize each cluster (LLM) → store summary text for BM25 and system prompt evidence

```sql
CREATE TABLE graph_community (
  id              uuid PRIMARY KEY,
  conversation_id uuid NOT NULL,
  name            text NOT NULL,
  summary         text NOT NULL,
  -- centroid embedding (incrementally maintained from member entity embeddings)
  centroid        vector(1024) NOT NULL,
  -- optional: embedding of (name + summary) for query routing (better recall than centroid for some queries)
  summary_embedding vector(1024) NULL,
  member_count    integer NOT NULL DEFAULT 0,
  dirty           boolean NOT NULL DEFAULT true,
  member_hash     text NOT NULL DEFAULT '',
  hit_count_7d    integer NOT NULL DEFAULT 0,
  created_at      timestamptz NOT NULL DEFAULT now(),
  updated_at      timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_graph_community_embedding_hnsw
  ON graph_community USING hnsw (centroid vector_ip_ops);

CREATE INDEX IF NOT EXISTS idx_graph_community_summary_embedding_hnsw
  ON graph_community USING hnsw (summary_embedding vector_ip_ops);

CREATE TABLE graph_community_member (
  conversation_id uuid NOT NULL,
  community_id    uuid NOT NULL REFERENCES graph_community(id) ON DELETE CASCADE,
  entity_id       uuid NOT NULL REFERENCES graph_entity(id) ON DELETE CASCADE,
  membership      real NOT NULL DEFAULT 1.0,
  created_at      timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (community_id, entity_id)
);
```

For a true hierarchy, add `graph_community_edge(parent_id, child_id)` and traverse with recursive CTE similarly.

---

## Graph construction (keep it cheap)

### Principle: reuse what Plast Mem already computes

Plast Mem already produces:

- semantic facts with `keywords: TEXT[]`
- provenance `source_episodic_ids: UUID[]`

So the *simplest* high-impact graph build is:

1) **Entities** = `semantic_memory.keywords`
2) **Fact links**: entity ↔ semantic fact (`graph_entity_fact`)
3) **Episode links**: entity ↔ episodes in `source_episodic_ids` (`graph_entity_episode`)
4) **Association edges**: connect keyword pairs within each fact (`cooccur_fact`)

This avoids new LLM calls and tends to lift “multi-hop / indirect cue” questions.

### Optional boosts (still simple)

#### Integrate a cheap extractor into `keywords` generation (recommended)

To keep the implementation simple, treat Graph as a **derived index** from semantic facts. That means the best ROI is to make `semantic_memory.keywords` stronger.

Update the keywords generator (currently a whitespace splitter in `PredictCalibrateJob`) to extract keywords from:

- the fact `statement`
- the source episode `title + summary` (adds named entities and bridge tokens)

Cheap-but-effective extraction targets:

- Code/tool tokens: `[A-Za-z][A-Za-z0-9_+./:-]{1,48}` (e.g., `pgvector`, `Graphiti`, `tokio`, `foo/bar`)
- Versions/models: `v?\d+(\.\d+){1,3}`, `gpt-4.1`, `qwen3:4b`
- CJK entity chunks: `[\p{Han}]{2,8}` with a small stopword filter
- Quoted/book-title phrases (short)

Keep `keywords` at **8–12 unique normalized tokens**. This improves both:

- BM25 on `semantic_memory.search_text`
- graph entity coverage (reduces fragmentation and dead-ends)

#### Canonicalization / aliasing (recommended)

Keep entity merging simple and deterministic:

- upsert entities by `(conversation_id, normalized)` first
- optionally add `alias` edges when two canonicals are merged
- avoid expensive global entity linking; this is per-`conversation_id` only

- Canonicalization:
  - store multiple surface forms as `alias` edges
  - unify by (normalized text exact match) first, then trigram similarity, then embedding similarity

### Where to hook in code

1) After semantic consolidation writes/updates facts:
   - in `crates/worker/src/jobs/predict_calibrate.rs` after `insert_new_fact()` / merge/update actions
2) After episodic creation:
   - in `crates/worker/src/jobs/event_segmentation.rs` after episode insert, optionally extract entities

Implementation-wise, put ingestion logic in **core** (so both worker + server can call it):

- `crates/core/src/memory/graph.rs` (new): `GraphMemory::ingest_*` + `GraphMemory::retrieve_*`

---

## Retrieval: recursive CTE traversal (k-hop) + scoring

### When to run the graph route (do not always-on)

Graph traversal is powerful but noisy. Run it only when needed:

- the query includes multi-hop/bridging signals (pronouns, “their teammate”, “that project”, “之前提到的”, etc.)
- Route 1 retrieval is low-confidence (few hits or low top scores)

When triggered, graph retrieval acts as **second-stage candidate generation**, then fuses back into Route 1 via RRF.

### 1) Seed entities from query (BM25 + vector RRF)

Use the same pattern as semantic/episodic retrieval:

- BM25 via `search_text ||| $query` and `pdb.score(id)`
- Vector via `embedding <#> $query_embedding`
- Fuse with RRF

### 1.5) Community-first seeding (Option B, closer to Mnemis)

To approximate Graphiti/Mnemis “top-down routing”:

1) retrieve top communities (BM25 on `name+summary`, and/or vector on `summary_embedding` or `centroid`)
2) collect seed entities from community membership
3) union with query-seeded entities

Default balanced parameters:

- `top_communities C = 2`
- `entities_per_community M = 40`
- direct query entity seeds `E = 15`

### 2) Recursive expansion (deliberate graph walk)

Core idea: expand from seed entities to neighbors with attenuation.

```sql
WITH RECURSIVE
seed AS (
  -- entity_id, seed_score already computed via RRF
  SELECT entity_id, seed_score::float8 AS score, 0 AS depth
  FROM seed_entities
),
walk AS (
  SELECT entity_id, score, depth
  FROM seed

  UNION ALL

  SELECT
    e.dst_entity_id AS entity_id,
    (w.score * e.weight * $HOP_DECAY)::float8 AS score,
    w.depth + 1 AS depth
  FROM walk w
  JOIN graph_edge e
    ON e.conversation_id = $CONV
   AND e.invalid_at IS NULL
   AND e.src_entity_id = w.entity_id
  WHERE w.depth < $MAX_DEPTH
),
entity_rank AS (
  SELECT entity_id, MAX(score) AS score
  FROM walk
  GROUP BY entity_id
  ORDER BY score DESC
  LIMIT $ENTITY_LIMIT
)
SELECT * FROM entity_rank;
```

Notes:

- Keep `MAX_DEPTH` small (2–3) for latency.
- Default balanced config: `MAX_DEPTH = 2`, `HOP_DECAY = 0.6`
- Keep edge fanout bounded by:
  - filtering `edge_type IN (...)`
  - optionally only expanding top-N edges per node (precomputed rank column or a partial index strategy).
  - Default balanced config: fanout ≤ 20 per node, edge types: `cooccur_fact`, `cooccur_episode`, `alias`

### 3) Convert entities → facts/episodes and fuse with Route 1

```sql
-- facts
SELECT f.id, MAX(er.score * ef.weight) AS score
FROM entity_rank er
JOIN graph_entity_fact ef
  ON ef.conversation_id = $CONV AND ef.entity_id = er.entity_id
JOIN semantic_memory f
  ON f.id = ef.semantic_id
WHERE f.invalid_at IS NULL
GROUP BY f.id
ORDER BY score DESC
LIMIT $K;
```

Do the same for episodic memories via `graph_entity_episode`, then apply FSRS rerank as usual.

Finally merge:

- `Route 1 semantic` + `graph semantic`
- `Route 1 episodic` + `graph episodic`

using RRF to reduce tuning sensitivity.

### Evidence paths (recommended)

Carry a compact path witness in the recursive CTE (e.g., `entity_id[]` and `edge_type[]`) and keep at most 1–2 best paths per final hit.
This is often disproportionately helpful for long-horizon benchmarks because it lets the downstream answering model “see the bridge” instead of guessing.

---

## Expected quality vs simplicity (recommendation)

### Option A (ultra-simple, likely **below** Mnemis)

- Build graph only from `semantic_memory.keywords` + `source_episodic_ids`
- No communities/hierarchy

Pros: minimal code, no extra LLM calls.
Cons: less “global selection”; limited when semantic extraction misses entities.

### Option B (recommended, closest to Mnemis with modest extra complexity)

Option A +:

- community nodes (one-level “topic” buckets) with embeddings + summaries
- dual-route retrieval: entity walk + community-first recall

This is still compact (a few more tables + a periodic worker job) and most directly targets LoCoMo/LongMemEval failure modes.

#### Option B minimal closed-loop defaults (quality/efficiency balance)

Community maintenance (rule B):

- assign entity → nearest community by vector similarity (top-10 search)
- join threshold: cosine ≥ **0.82**, else create new community
- maintain `centroid` as incremental mean of member entity embeddings
- low-frequency merge: cosine ≥ **0.90** (small into large), mark dirty

LLM summarization (cost-controlled):

- only summarize dirty communities that are “important” (e.g., `member_count ≥ 8` or top by `hit_count_7d`)
- per run: summarize at most **5** communities
- write `name` + `summary` and (optionally) `summary_embedding = embed(name || ' ' || summary)`

Graph-route retrieval (only when triggered):

- community-first seeds: `C=2`, `M=40`, plus direct entity seeds `E=15`
- traversal: depth **2**, hop decay **0.6**, fanout ≤ **20**, edge types limited
- fuse with Route 1 using RRF (episodic final score still × FSRS)

---

## Performance checklist (PostgreSQL)

- Keep everything scoped by `conversation_id` (btree indexes must include it).
- HNSW on entity/community embeddings, ParadeDB BM25 on entity search_text.
- Recursive CTE depth ≤ 3; seed size ≤ 50; expand only edge types you need.
- Prefer “graph route” as a **second-stage** augmenter (fuse with Route 1) instead of always-on heavy traversal.
