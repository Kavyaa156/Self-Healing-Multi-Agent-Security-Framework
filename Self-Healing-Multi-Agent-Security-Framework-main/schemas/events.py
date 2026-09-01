import time
from typing import Optional, Any, Dict, List, Tuple
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# P1 Emits Raw Event Data
# ---------------------------------------------------------------------------
class RawEvent(BaseModel):
    task_id: str
    agent_id: str
    step_id: int
    event_type: str  # e.g., "thought", "tool_call", "tool_output", "final_answer"
    content: str
    tool_name: Optional[str] = None
    tool_input: Optional[Dict[str, Any]] = None
    tool_output: Optional[Dict[str, Any]] = None
    success: bool = True
    timestamp: float = Field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Shared Data Contract Across P2, P3, P4
# ---------------------------------------------------------------------------
class ExecutionEvent(BaseModel):
    task_id: str
    agent_id: str
    step_id: int
    event_type: str
    content: str
    tool_name: Optional[str] = None
    tool_input: Optional[Dict[str, Any]] = None
    tool_output: Optional[Dict[str, Any]] = None
    success: bool = True
    timestamp: float = Field(default_factory=time.time)


class ReliabilityScore(BaseModel):
    task_id: str
    step_id: int
    C: float
    S: float
    E: float
    R: float
    weights: Tuple[float, float, float] = (0.4, 0.4, 0.2)
    threshold: float = 0.65


class FailureDiagnosis(BaseModel):
    task_id: str
    step_id: int
    failure_type: str  # "F1", "F2", "F3", or "F4"
    root_cause_agent: str
    description: str
    reliability_score: ReliabilityScore
    # --- Phase 5 Novelty #1 additions (optional, backward compatible) ---
    root_cause_step: Optional[int] = None
    propagation_chain: Optional[List[Dict[str, Any]]] = None
    attribution_confidence: Optional[float] = None


class TaskSpec(BaseModel):
    task_id: str
    description: str
    initial_input: Dict[str, Any]

class FailureAttribution(BaseModel):
    """
    Phase 5 Novelty #1 output: full root-cause trace across the
    ExecutionGraph -- not just "F4 or not", but WHICH agent/step
    actually originated the failure, plus the evidence chain.
    """
    root_cause_agent: str
    root_cause_step: int
    propagation_chain: List[Dict[str, Any]]
    confidence: float = 1.0