"""
Person 3 — Reliability Evaluation Module
validation_seed_data.py — domain-reference chunks for the Semantic
Accuracy (S) validation store.

Per the plan's [PROJECT DESIGN DECISION] in Section 4: "Task domain
(needed for P3's validation database) is not specified by Jeong — team
must pick and document this explicitly, since it determines what
'domain-relevant validation database' (S) checks against."

Domain chosen: security / access-control auditing. This is inferred
directly from the team's own repo name
(Self-Healing-Multi-Agent-Security-Framework) and the real sample task
in telemetry_events.jsonl (task_sec_101 — an Authentication Gateway
privilege-escalation audit). If the team later locks in a different or
broader task domain, only this file needs to change — nothing in
semantic_accuracy.py depends on this specific content.

Each entry is a short, single-idea reference statement (not a full
policy document) so that Chroma's nearest-neighbor retrieval can match
an event's content against a specific, verifiable claim rather than a
long mixed-topic paragraph.
"""

SECURITY_AUDIT_REFERENCE_DOCS: list[str] = [
    # Least privilege / default permissions
    "New user accounts should be provisioned with least-privilege "
    "defaults, granting only the minimum access required, not broad "
    "read or write permissions by default.",
    "Granting write access to configuration files by default violates "
    "the principle of least privilege and increases the risk of "
    "accidental or malicious configuration changes.",

    # Role hierarchy / privilege escalation
    "Role hierarchies must not allow implicit inheritance of "
    "administrator-level permissions by lower-privileged roles such as "
    "manager or staff roles.",
    "A user granted a mid-level role should never be able to perform "
    "admin-level actions such as modifying user policies or accessing "
    "audit logs, unless explicitly and separately authorized.",

    # Authentication / lockout policy
    "Account lockout policies should trigger after a small number of "
    "consecutive failed login attempts, typically five or fewer, to "
    "reduce the window for credential-stuffing attacks.",
    "A high volume of failed login attempts from a single source "
    "followed by a successful login is a strong indicator of "
    "credential-stuffing or brute-force compromise.",

    # API / role-assignment security
    "Any API endpoint that assigns or elevates a user's role must "
    "validate the caller's token scope before completing the "
    "assignment, to prevent unauthorized privilege escalation.",
    "Accepting role-assignment requests without verifying an "
    "authentication token allows any holder of a valid token to "
    "self-grant elevated privileges such as administrator access.",

    # Monitoring / audit trail
    "Continuous monitoring and anomaly detection on role-grant and "
    "privilege-change events provides early warning of potential "
    "privilege-escalation attempts.",
    "Audit logs of privilege-grant events should be reviewed "
    "regularly to detect unauthorized or unusual permission changes.",

    # Remediation process
    "Identified access-control misconfigurations should be prioritized "
    "by risk level and assigned a clear owner and target completion "
    "date for remediation.",
    "Security remediation changes should be version-controlled, "
    "reviewed via pull request, and deployed through a CI/CD pipeline "
    "with automated rollback on failure.",
]
