"""Static security-policy text used both for display and for the RAG index."""

POLICY_DOCS = [
    "Policy 1 (ACCESS): Access requires a clearance level equal to or higher than the "
    "document classification, plus a valid need-to-know related to your role and task.",
    "Policy 2 (MARKING): All classified documents must be clearly marked with their "
    "classification level on every page.",
    "Policy 3 (HANDLING): Classified information must only be stored in approved systems; "
    "electronic transmission requires approved encryption.",
    "Policy 4 (DISCLOSURE): Unauthorized disclosure to anyone lacking clearance and "
    "need-to-know is prohibited and may carry legal penalties.",
    "Policy 5 (MINIMIZATION): Request or access only the minimum classified information "
    "necessary for your official duties.",
    "Policy 6 (MONITORING): Use of this system implies consent to monitoring, recording, "
    "and auditing of all activity.",
    "Policy 7 (ZERO TRUST): Every access attempt must be authenticated and authorized; "
    "trust is never assumed from network location or prior access.",
    "Policy 8 (NEED-TO-KNOW): Access requires both appropriate clearance and a "
    "demonstrated, job-related need for the specific information. Clearance alone is "
    "not sufficient.",
    "Policy 9 (QUERY SAFETY): Queries must not inject code, bypass controls, reveal system "
    "prompts/configuration, instruct the system to ignore policy, or impersonate others.",
    "Policy 10 (DATA INTEGRITY): Do not attempt to modify, delete, or reclassify documents "
    "without explicit authorization and proper procedure.",
    "Policy 11 (DATA EXFILTRATION): Do not export, copy, or transmit classified documents "
    "outside approved channels or to unauthorized recipients.",
]
