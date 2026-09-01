import networkx as nx
from typing import Dict, List, Optional
from schemas.events import ExecutionEvent


class ExecutionGraph:
    """
    Wraps a networkx.DiGraph representing one or more tasks' execution.

    - Node  = one ExecutionEvent (a single step)
    - Edge  = execution order (previous step -> this step, within the same task_id)

    Each node also gets stamped with two anomaly flags at the moment it's
    added, computed as of that point in the execution:
      - repeated_failure   : this agent has failed 3+ tool_calls in a row
      - abnormal_sequence  : this task has revisited a step_id it already ran
    """

    def __init__(self):
        self.graph = nx.DiGraph()
        self._node_counter = 0                     # global chronological order
        self._last_node: Dict[str, str] = {}        # task_id -> most recent node_id
        self._seen_steps: Dict[str, List[int]] = {} # task_id -> step_ids seen so far

    def add_event(self, event: ExecutionEvent) -> None:
        # Track whether this step_id has already occurred for this task
        # BEFORE recording the current one (so a revisit is detected correctly).
        seen = self._seen_steps.setdefault(event.task_id, [])
        is_revisit = event.step_id in seen
        seen.append(event.step_id)

        self._node_counter += 1
        node_id = f"{event.task_id}:{event.step_id}:{self._node_counter}"

        self.graph.add_node(
            node_id,
            order=self._node_counter,
            task_id=event.task_id,
            step_id=event.step_id,
            agent_id=event.agent_id,
            event_type=event.event_type,
            success=event.success,
        )

        prev_node = self._last_node.get(event.task_id)
        if prev_node is not None:
            self.graph.add_edge(prev_node, node_id)
        self._last_node[event.task_id] = node_id

        # Run both pattern checks now that the node exists, and stamp the
        # result onto the node itself (useful for debugging / paper figures).
        repeated_failure = self.get_repeated_failures(event.agent_id)
        abnormal_sequence = is_revisit or self.get_abnormal_sequence(event.task_id)
        self.graph.nodes[node_id]["repeated_failure"] = repeated_failure
        self.graph.nodes[node_id]["abnormal_sequence"] = abnormal_sequence

    def get_repeated_failures(self, agent_id: str, window: int = 3) -> bool:
        """
        True if `agent_id` has `window` or more CONSECUTIVE failed tool_call
        events (looking across all tasks/steps seen so far, in chronological order).
        """
        nodes = [
            n for n, d in self.graph.nodes(data=True)
            if d["agent_id"] == agent_id and d["event_type"] == "tool_call"
        ]
        nodes.sort(key=lambda n: self.graph.nodes[n]["order"])

        streak = 0
        for n in nodes:
            if not self.graph.nodes[n]["success"]:
                streak += 1
                if streak >= window:
                    return True
            else:
                streak = 0
        return False

    def get_abnormal_sequence(self, task_id: str) -> bool:
        """True if this task has executed the same step_id more than once (a loop)."""
        seen = self._seen_steps.get(task_id, [])
        return len(seen) != len(set(seen))

    def get_upstream_failed(self, task_id: str, step_id: int) -> bool:
        """
        Return True if any ancestor step in the same task has failed.

        This is used by P4 to detect an F4 (Workflow Propagation Error):
        the current step may appear successful, but an earlier upstream
        step in the workflow has already failed.
        """
        target = self._find_node(task_id, step_id)
        if target is None:
            return False

        ancestors = nx.ancestors(self.graph, target)

        return any(
            not self.graph.nodes[a]["success"]
            for a in ancestors
        )

    def get_flags(self, task_id: str, agent_id: str = None) -> Dict[str, bool]:
        """
        Convenience for P4: both flags in one call.
        Pass agent_id to also check that specific agent's repeated-failure streak;
        otherwise only abnormal_sequence is checked.
        """
        return {
            "repeated_failure": self.get_repeated_failures(agent_id) if agent_id else False,
            "abnormal_sequence": self.get_abnormal_sequence(task_id),
        }

    def _find_node(self, task_id: str, step_id: int) -> "Optional[str]":
        """Return the most recent node_id matching (task_id, step_id), or None."""
        target = None
        for n, d in self.graph.nodes(data=True):
            if d["task_id"] == task_id and d["step_id"] == step_id:
                if target is None or d["order"] > self.graph.nodes[target]["order"]:
                    target = n
        return target

    def get_failed_ancestors_chronological(self, task_id: str, step_id: int) -> List[Dict]:
        """
        Return node-data dicts for ALL failed ancestors of the given step,
        sorted chronologically (earliest execution order first).

        This is the core of trajectory-based attribution: the failure
        that happened FIRST in time is the plausible true origin, not
        necessarily the one structurally closest (immediate parent) to
        the step where the failure surfaced.
        """
        target = self._find_node(task_id, step_id)
        if target is None:
            return []
        ancestors = nx.ancestors(self.graph, target)
        failed = [
            {**self.graph.nodes[a], "node_id": a}
            for a in ancestors
            if not self.graph.nodes[a]["success"]
        ]
        failed.sort(key=lambda d: d["order"])
        return failed

    def get_root_cause_node(self, task_id: str, step_id: int) -> "Optional[Dict]":
        """
        Return the EARLIEST failed ancestor of the given step -- the true
        origin of a propagating failure -- or None if no ancestor failed.
        """
        failed = self.get_failed_ancestors_chronological(task_id, step_id)
        return failed[0] if failed else None

    def get_propagation_chain(self, task_id: str, step_id: int) -> List[Dict]:
        """
        Return the ordered evidence chain: every failed ancestor (earliest
        first) followed by the target step itself.
        """
        target = self._find_node(task_id, step_id)
        if target is None:
            return []
        target_data = {**self.graph.nodes[target], "node_id": target}
        failed_ancestors = self.get_failed_ancestors_chronological(task_id, step_id)
        if not failed_ancestors:
            return [target_data]
        return failed_ancestors + [target_data]