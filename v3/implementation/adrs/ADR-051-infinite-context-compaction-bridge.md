# ADR-051: Infinite Context via Compaction-to-Memory Bridge

**Status:** Proposed
**Date:** 2026-02-10
**Authors:** RuvNet, Claude Flow Team
**Version:** 1.0.0
**Related:** ADR-006 (Unified Memory), ADR-009 (Hybrid Memory Backend), ADR-027 (RuVector PostgreSQL), ADR-048 (Auto Memory Integration), ADR-049 (Self-Learning Memory GNN)

## Context

### The Problem: Context Window is a Hard Ceiling

Claude Code operates within a finite context window. When the conversation approaches
this limit, the system automatically **compacts** prior messages -- summarizing them
into a condensed form. While compaction preserves the gist of the conversation, it
irreversibly discards:

- **Tool call details**: Exact file paths edited, bash commands run, grep results
- **Decision reasoning**: Why a particular approach was chosen over alternatives
- **Code context**: Specific code snippets discussed, error messages diagnosed
- **Multi-step workflows**: The sequence of operations that led to a result
- **Agent coordination state**: Swarm agent outputs, task assignments, memory keys

This creates a "context cliff" -- once compaction occurs, Claude loses the ability to
reference specific earlier details, leading to repeated work, lost context, and
degraded assistance quality in long sessions.

### What We Have Today

Claude Code's SDK exposes two hook events relevant to compaction:

1. **PreCompact** (`PreCompactHookInput`): Fires BEFORE compaction with access to:
   - `transcript_path`: Full JSONL transcript of the conversation
   - `session_id`: Current session identifier
   - `trigger`: `'manual'` or `'auto'`
   - `custom_instructions`: Optional compaction guidance

2. **SessionStart** (`SessionStartHookInput`): Fires AFTER compaction with:
   - `source: 'compact'` (distinguishes post-compaction from fresh start)
   - Hook output supports `additionalContext` injection into the new context

Current PreCompact hooks (`.claude/settings.json` lines 469-498) only:
- Print guidance text about available agents
- Export learned patterns to `compact-patterns.json`
- Export intelligence state to `intelligence-state.json`

**They do NOT capture the actual conversation content.** After compaction, the rich
transcript is gone.

### What We Want

An "infinite context" system where:
1. Before compaction, conversation turns are chunked, summarized, embedded, and stored
   in the AgentDB/RuVector memory backend
2. After compaction, the most relevant stored context is retrieved and injected back
   into the new context window via `additionalContext`
3. Across sessions, accumulated transcript archives enable cross-session context
   retrieval -- Claude can recall details from previous conversations

## Decision

Implement a **Compaction-to-Memory Bridge** as a hook script that intercepts the
PreCompact lifecycle and stores conversation history in the AgentDB memory backend
(with optional RuVector PostgreSQL scaling). On post-compaction SessionStart, the
bridge retrieves and injects the most relevant archived context.

### Design Principles

1. **Hook-Native**: Uses Claude Code's official PreCompact and SessionStart hooks --
   no monkey-patching, no SDK modifications
2. **Backend-Agnostic**: Works with JsonFileBackend (zero dependencies), AgentDB
   (HNSW vectors), or RuVector PostgreSQL (TB-scale) -- graceful degradation
3. **Timeout-Safe**: All operations complete within the 5-second hook timeout using
   local I/O and hash-based embeddings (no LLM calls, no network)
4. **Dedup-Aware**: Content hashing prevents re-storing on repeated compactions
5. **Budget-Constrained**: Restored context fits within a configurable character
   budget (default 4000 chars) to avoid overwhelming the new context window
6. **Non-Blocking**: Hook failures are silently caught -- compaction always proceeds

## Architecture

### System Context

```
+------------------------------------------------------------------+
|                      Claude Code Session                          |
|                                                                   |
|  Context Window: [system prompt] [messages...] [new messages]     |
|                                                                   |
|  +----------------------+                                         |
|  | Context Window Full  |                                         |
|  | PreCompact fires     |-----------------------------+           |
|  +----------------------+                             |           |
|                                                       v           |
|  +-----------------------------------------------------------+   |
|  |           context-persistence-hook.mjs                     |   |
|  |                                                            |   |
|  |  1. Read transcript_path (JSONL)                           |   |
|  |  2. Parse -> filter -> chunk by turns                      |   |
|  |  3. Extract summaries + metadata                           |   |
|  |  4. Generate hash embeddings (768-dim)                     |   |
|  |  5. Store as episodic MemoryEntry[]                        |   |
|  |     namespace: 'transcript-archive'                        |   |
|  +---------------------------+--------------------------------+   |
|                              |                                    |
|                              v                                    |
|  +-----------------------------------------------------------+   |
|  |              Memory Backend (tiered)                        |   |
|  |                                                            |   |
|  |  Tier 1: JsonFileBackend                                   |   |
|  |    -> .claude-flow/data/transcript-archive.json            |   |
|  |    -> Zero dependencies, always available                  |   |
|  |                                                            |   |
|  |  Tier 2: AgentDB + HNSW  (if @claude-flow/memory built)   |   |
|  |    -> 150x-12,500x faster semantic search                  |   |
|  |    -> Vector-indexed retrieval                             |   |
|  |                                                            |   |
|  |  Tier 3: RuVector PostgreSQL (if configured)               |   |
|  |    -> TB-scale storage                                     |   |
|  |    -> GNN-enhanced retrieval                               |   |
|  |    -> Self-learning query optimizer                        |   |
|  +-----------------------------------------------------------+   |
|                                                                   |
|  +----------------------+                                         |
|  | Compaction complete   |                                        |
|  | SessionStart fires   |-----------------------------+           |
|  | source: 'compact'    |                             |           |
|  +----------------------+                             v           |
|                                                                   |
|  +-----------------------------------------------------------+   |
|  |           context-persistence-hook.mjs                     |   |
|  |                                                            |   |
|  |  1. Detect source === 'compact'                            |   |
|  |  2. Query transcript-archive for session_id                |   |
|  |  3. Rank by recency, fit within char budget                |   |
|  |  4. Return { additionalContext: "..." }                    |   |
|  +-----------------------------------------------------------+   |
|                                                                   |
|  New Context Window: [system] [compact summary] [restored ctx]    |
|                      [new messages continue...]                   |
+-------------------------------------------------------------------+
```

### Transcript Parsing

The `transcript_path` is a JSONL file where each line is an `SDKMessage`:

| Message Type | Content | Action |
|-------------|---------|--------|
| `user` | `message.content[]` (text blocks, tool_result blocks) | **Extract**: user prompts, tool results |
| `assistant` | `message.content[]` (text blocks, tool_use blocks) | **Extract**: responses, tool calls with inputs |
| `result` | Session summary, usage stats | **Extract**: cost, turn count |
| `system` (init) | Tools, model, MCP servers | **Skip** (not conversation content) |
| `stream_event` | Partial streaming data | **Skip** (redundant with complete messages) |
| `tool_progress` | Elapsed time updates | **Skip** |

### Chunking Strategy

Messages are grouped into **conversation turns**:

```
Chunk N = {
  userMessage: SDKUserMessage,
  assistantMessage: SDKAssistantMessage,
  toolCalls: [
    { name: 'Edit', input: { file_path: '...' } },
    { name: 'Bash', input: { command: '...' } },
  ],
  metadata: {
    toolNames: ['Edit', 'Bash'],
    filePaths: ['/src/foo.ts'],
    turnIndex: N,
    timestamp: '...',
  }
}
```

**Boundary rules:**
- New user message (non-synthetic) = new chunk
- Cap at last 500 messages for timeout safety
- Skip synthetic user messages (tool result continuations)

### Summary Extraction (No LLM)

For each chunk, extractive summarization:

```
Summary = [
  firstLine(userMessage.text),
  "Tools: " + toolNames.join(", "),
  "Files: " + filePaths.join(", "),
  firstTwoLines(assistantMessage.text),
].join(" | ").slice(0, 300)
```

### Memory Entry Schema

```typescript
{
  key: `transcript:${sessionId}:${chunkIndex}:${timestamp}`,
  content: fullChunkText,
  type: 'episodic',
  namespace: 'transcript-archive',
  tags: ['transcript', 'compaction', sessionId, ...toolNames],
  metadata: {
    sessionId: string,
    chunkIndex: number,
    trigger: 'manual' | 'auto',
    timestamp: string,
    toolNames: string[],
    filePaths: string[],
    summary: string,
    contentHash: string,
    preTokens: number,
    turnRange: [start, end],
  },
  accessLevel: 'private',
}
```

### Context Restoration

On `SessionStart(source: 'compact')`:

1. Query `transcript-archive` namespace for `metadata.sessionId === current_session`
2. Also query for cross-session entries with similar tool/file patterns (future)
3. Sort by `chunkIndex` descending (most recent first)
4. Build restoration text fitting within char budget
5. Return via `hookSpecificOutput.additionalContext`

### Hash Embedding Function

Reused from `learning-bridge.ts:425-450` (deterministic, sub-millisecond):

```javascript
function createHashEmbedding(text, dimensions = 768) {
  const embedding = new Float32Array(dimensions);
  const normalized = text.toLowerCase().trim();
  for (let i = 0; i < dimensions; i++) {
    let hash = 0;
    for (let j = 0; j < normalized.length; j++) {
      hash = ((hash << 5) - hash + normalized.charCodeAt(j) * (i + 1)) | 0;
    }
    embedding[i] = (Math.sin(hash) + 1) / 2;
  }
  let norm = 0;
  for (let i = 0; i < dimensions; i++) norm += embedding[i] * embedding[i];
  norm = Math.sqrt(norm);
  if (norm > 0) for (let i = 0; i < dimensions; i++) embedding[i] /= norm;
  return embedding;
}
```

## Performance Budget

| Operation | Time Budget | Actual |
|-----------|------------|--------|
| Read stdin (hook input) | 100ms timeout | <10ms |
| Read transcript JSONL | 500ms | ~50ms for 500 messages |
| Parse + filter messages | 200ms | ~20ms |
| Chunk + extract summaries | 200ms | ~30ms |
| Generate hash embeddings | 100ms | <1ms total |
| Content hash (SHA-256) | 100ms | <5ms |
| Store to JsonFileBackend | 500ms | ~50ms |
| **Total (PreCompact)** | **5000ms** | **~170ms** |
| Query + build context | 500ms | ~30ms |
| **Total (SessionStart)** | **6000ms** | **~40ms** |

## Security Considerations

1. **No credentials in transcript**: Tool inputs may contain file paths but not secrets
   (Claude Code already redacts sensitive content before tool execution)
2. **Local storage only**: JsonFileBackend writes to `.claude-flow/data/` which is
   gitignored. No network calls in the hook path.
3. **No SQL injection risk**: JsonFileBackend uses file I/O, not SQL. When AgentDB
   is used, it's an in-memory database with typed APIs.
4. **Content hashing**: Uses `crypto.createHash('sha256')` for dedup -- standard Node.js
5. **Graceful failure**: All operations wrapped in try/catch. Hook failures produce
   empty output -- compaction always proceeds normally.

## Migration Path

### Phase 1: JsonFileBackend (Immediate)
- Zero dependencies, works everywhere
- Stores transcript chunks as JSON
- Retrieval by namespace + metadata filter
- No semantic search (recency-based only)

### Phase 2: AgentDB Integration (When built)
- If `@claude-flow/memory` dist exists, use `AgentDBBackend`
- HNSW-indexed embeddings enable semantic search across archived transcripts
- Cross-session retrieval: "What did we discuss about auth?" finds relevant chunks

### Phase 3: RuVector PostgreSQL (When configured)
- TB-scale transcript archives
- GNN-enhanced retrieval using code dependency graphs
- Self-learning query optimizer tunes retrieval over time
- Multi-user/multi-session search across team archives

## Consequences

### Positive

1. **No more context cliff**: Conversation details survive compaction as structured
   memory entries with semantic embeddings
2. **Cross-session recall**: Archived transcripts accumulate across sessions, enabling
   "What did we do last time?" queries
3. **Tiered scaling**: Works with zero dependencies (JSON) up to TB-scale (RuVector)
4. **Non-invasive**: Uses official SDK hooks -- no patches, no internal API dependencies
5. **Composable**: Transcript entries in AgentDB are searchable alongside patterns,
   learnings, and other memory types

### Negative

1. **Storage growth**: Long sessions produce many chunks. Mitigation: configurable
   retention policy, namespace-level cleanup
2. **Summary quality**: Extractive summarization is fast but imprecise. Mitigation:
   full content is stored; summaries are just for the restoration preview
3. **No semantic search in Tier 1**: JsonFileBackend can only filter by metadata.
   Mitigation: AgentDB upgrade enables real vector search

### Neutral

1. **Hook timeout pressure**: 5s budget is generous for local I/O operations
2. **Embedding quality**: Hash embeddings are deterministic approximations. When real
   ONNX embeddings are available, they can replace hash embeddings transparently

## Future Enhancements

1. **Semantic restoration**: Use real embeddings + the user's first post-compaction
   message to query the most RELEVANT archived chunks (not just most recent)
2. **Compaction summary capture**: After compaction, store Claude's own summary as a
   high-confidence semantic entry alongside our chunk-level detail
3. **Retention policies**: Auto-expire old transcript archives, keep only chunks that
   were accessed (indicating relevance)
4. **Cross-session search MCP tool**: Expose `transcript-archive` search as an MCP
   tool so Claude can explicitly query past conversations
5. **MemoryGraph integration**: Add reference edges between sequential chunks for
   PageRank-aware retrieval (ADR-049)

## References

- ADR-006: Unified Memory Service
- ADR-009: Hybrid Memory Backend (AgentDB + SQLite)
- ADR-027: RuVector PostgreSQL Integration
- ADR-048: Auto Memory Integration
- ADR-049: Self-Learning Memory with GNN
- Claude Agent SDK: `@anthropic-ai/claude-agent-sdk` PreCompact hook types
