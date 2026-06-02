// seed/init_clpapers.js
// Runs automatically when MongoDB container starts for the first time.
// Creates the `clpapers` database and the `chunks` collection with
// an index on chunk_id, plus two sample documents so the notebook
// can run even before the ingestion pipeline (notebook 01) has loaded
// real papers.

db = db.getSiblingDB('clpapers');

// ── Create collection with schema validation ────────────────
db.createCollection('chunks', {
  validator: {
    $jsonSchema: {
      bsonType: 'object',
      required: ['chunk_id', 'paper_id', 'text'],
      properties: {
        chunk_id:   { bsonType: 'string' },
        paper_id:   { bsonType: 'string' },
        text:       { bsonType: 'string' },
        title:      { bsonType: 'string' },
        authors:    { bsonType: 'string' },
        year:       { bsonType: ['int', 'string'] },
        page_start: { bsonType: 'int' },
        page_end:   { bsonType: 'int' },
      }
    }
  },
  validationAction: 'warn'   // warn instead of error for flexibility
});

// ── Index on chunk_id (unique) ──────────────────────────────
db.chunks.createIndex({ chunk_id: 1 }, { unique: true });
db.chunks.createIndex({ paper_id: 1 });

// ── Sample seed documents ───────────────────────────────────
// These are placeholder chunks so the notebook doesn't crash on an
// empty collection. Replace / extend with your real ingestion pipeline.
db.chunks.insertMany([
  {
    chunk_id:   'sample_001_chunk_0',
    paper_id:   'sample_001',
    title:      'Attention Is All You Need',
    authors:    'Vaswani et al.',
    year:       2017,
    page_start: 1,
    page_end:   2,
    text:       'The dominant sequence transduction models are based on complex recurrent or convolutional neural networks. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.'
  },
  {
    chunk_id:   'sample_002_chunk_0',
    paper_id:   'sample_002',
    title:      'BERT: Pre-training of Deep Bidirectional Transformers',
    authors:    'Devlin et al.',
    year:       2019,
    page_start: 1,
    page_end:   2,
    text:       'We introduce BERT, a language representation model standing for Bidirectional Encoder Representations from Transformers. BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context.'
  },
  {
    chunk_id:   'sample_003_chunk_0',
    paper_id:   'sample_003',
    title:      'Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks',
    authors:    'Lewis et al.',
    year:       2020,
    page_start: 1,
    page_end:   2,
    text:       'We explore a general-purpose fine-tuning recipe for retrieval-augmented generation (RAG), combining parametric and non-parametric memory for language generation. RAG models retrieve relevant passages from Wikipedia using a dense retrieval component.'
  }
]);

print('✅  clpapers DB seeded: ' + db.chunks.countDocuments() + ' sample chunks inserted.');
