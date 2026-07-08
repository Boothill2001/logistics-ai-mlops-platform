# Architecture

## System overview

```mermaid
flowchart LR
    subgraph Clients
        OPS[Ops / Sales / Intern]
    end

    subgraph API["FastAPI service"]
        MW[Request-ID middleware<br/>structured logging]
        PRED[POST /predict/delay]
        CHAT[POST /copilot/chat]
        APPR[POST /copilot/approve]
        HEALTH[/health /ready /metrics/]
    end

    subgraph ML["ML platform"]
        REG[(Model registry<br/>v1 production / v2 canary)]
        CANARY{Canary router<br/>hash 90/10}
        BATCH[Nightly batch scoring]
        DRIFT[PSI drift detector]
    end

    subgraph Copilot["LangGraph copilot"]
        GRAPH[14-node graph<br/>LLM plans, backend executes]
        LLM[LLM provider<br/>DeepSeek / Mock fallback]
    end

    subgraph Data
        CHROMA[(Chroma vector store<br/>permission metadata)]
        CSV[(Shipment CSVs)]
        AUDIT[(Audit log JSONL)]
    end

    PROM[Prometheus]

    OPS --> MW --> PRED & CHAT & APPR
    PRED --> CANARY --> REG
    CHAT --> GRAPH
    GRAPH --> LLM
    GRAPH -->|permission-filtered| CHROMA
    GRAPH -->|internal call| CANARY
    GRAPH --> CSV
    GRAPH --> AUDIT
    BATCH --> REG
    BATCH --> DRIFT
    BATCH --> CSV
    PROM -->|scrape /metrics| HEALTH
```

## Copilot graph

```mermaid
flowchart TD
    IV[InputValidationNode] -->|valid| IC[IntentClassificationNode]
    IV -->|invalid| FR
    IC --> PP[PermissionPolicyNode]
    PP -->|denied| FR[FinalResponseNode]
    PP -->|allowed| PL[PlannerNode] --> RD[RouterDecisionNode]

    RD -->|knowledge_search| RAG[RAGRetrievalNode]
    RD -->|ml_prediction| ML[MLInferenceNode]
    RD -->|batch_query| BQ[BatchQueryNode]
    RD -->|dangerous_action| ED[EmailDraftNode] --> HA[HumanApprovalNode]
    RD -->|missing_info| FR

    RAG & ML & BQ & HA --> OC[ObservationCheckNode]
    OC -->|result missing, first time| RP[ReplanNode] --> RD
    OC -->|ok| FR
    FR --> LA[LoggingAuditNode] --> E((END))
```

## Request lifecycles

**Real-time prediction**: request → Pydantic validation → canary router picks
v1 (90%) or v2 (10%) by hashing `shipment_id` → model from registry →
response with `model_version`, `latency_ms`, `request_id` → Prometheus
counters/histograms updated.

**Copilot question**: request + `X-User-Id` → validate → LLM classifies
intent (rule fallback) → customer-level RBAC gate → planner/router → executor
(retrieval is permission-filtered *inside* the vector query; predictions come
from the same predictor as the API; batch answers are row-level filtered) →
observation check with one replan → response with citations → audit entry.

**Dangerous action**: email is drafted by the LLM, stored as
`pending_approval`, and returned to the user. Nothing is ever sent by the
graph. Approval requires an ops/admin user on a separate endpoint, and even
approval only flips the draft's status — dispatch is out of scope by design.
