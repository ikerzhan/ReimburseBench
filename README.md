# ReimburseBench v1 — Full-Workflow Reimbursement Audit Benchmark for LMMs

ReimburseBench is a benchmark designed to evaluate the audit capabilities of large multimodal models (LMMs) in the domain of corporate reimbursement auditing. It tests whether an AI agent can:

1. **Analyze** multi-modal documents (receipt images, PDF policies, Excel spreadsheets, Word documents).
2. **Detect** compliance violations, forgeries, inconsistencies, and fraudulent intent.
3. **Compute** the maximum legally reimbursable amount (MAX_LEGAL) and determine a binary reimbursement flag (FLAG).
4. **Reason** step-by-step, citing specific policy clauses and evidence.

---

## Dataset Structure

```
ReimburseBench_v1/
├── infer.py                  # Inference script: builds prompts and calls LLM
├── eval.py                   # Evaluation script: scores FLAG, MAX_LEGAL, and logic points
├── eval_failure_mode.py      # Failure mode analysis
├── metadata.json             # Metadata for all workflows (difficulty, logic points)
├── workflow/                 # Contains ~151 sub-directories, one per workflow
│   ├── 1.1.1/
│   ├── 1.1.2/
│   ├── ...
│   └── 6.6.4/
└── README.md                 # This file
```

---

## Workflow Structure

Each subfolder under `workflow/` (e.g., `workflow/1.1.1/`) represents a single reimbursement audit case and contains:

### Task Description File

**`task_description.json`** — Defines the reimbursement request:

```json
{
    "id": "1.1.1",
    "context": "A researcher is claiming accommodation expenses for attending ISS 2024...",
    "required_amount": 1620.0,
    "currency": "USD",
    "source_files": {
        "Background_information": [
            "General Guidelines for Reimbursement of Travel Expenses...pdf",
            "Notice on Travel Arrangements for Participants...pdf",
            "Reimbursement_Standard.xlsx",
            "Attendee Itinerary Approval Form.xlsx"
        ],
        "Receipts": {
            "Li_Wei": "2_HotelBill_723.jpg"
        }
    }
}
```

| Field | Description |
|---|---|
| `id` | Unique workflow identifier |
| `context` | Natural language description of the reimbursement scenario |
| `required_amount` | The amount claimed by the applicant |
| `currency` | Currency code (e.g., USD, CNY, EUR) |
| `source_files.Background_information` | List of policy documents, guidelines, approvals, etc. |
| `source_files.Receipts` | Dictionary mapping attendee names to receipt file paths |

### Reference Answer File

**`reference_answer.json`** — The ground truth for evaluation:

```json
{
    "Scoring_point_1_Flag": 1,
    "Scoring_point_2_Max_legal": 1500.0,
    "Scoring_point_3_Problem_1": {
        "answer": "Amount-cap violation",
        "score": 5
    }
}
```

| Key | Description |
|---|---|
| `Scoring_point_1_Flag` | Binary flag: 1 = reimbursable, 0 = not fully reimbursable |
| `Scoring_point_2_Max_legal` | Maximum amount (float) that the policy allows |
| `Scoring_point_3_Problem_1` (and higher) | Content describing specific problems found in the case |

Each logic point contains:
- **`answer`**: A short description of the problem.
- **`score`**: The maximum score allocated to this point.


---

## Metadata

**`metadata.json`** is an array of all workflows with their difficulty levels and associated logic points:

```json
{
    "id": "1.1.1",
    "difficulty": "hard",
    "Scoring_point_3_Problem_1": ["Amount-cap violation"],
    "expected_types": ["Amount-cap violation"]
}
```

| Field | Description |
|---|---|
| `id` | Workflow identifier |
| `difficulty` | One of: `easy`, `medium`, `hard` |
| `Scoring_point_*` | Logic points expected for each scoring sub-question |
| `expected_types` | Full list of problem types expected in this workflow (across all scoring points, including the two workflow-level distractors) |

### Logic Point Categories

| English Name | Category | Description |
|---|---|---|
| Invoice content completeness | A. Evidence Integrity | Invoice missing required fields per policy |
| Supporting-document completeness | A. Evidence Integrity | Missing required supporting documents |
| Claimant identity matching | B. Cross-Doc Consistency | Person names / headcount mismatch |
| Invoice type matching | B. Cross-Doc Consistency | Invoice category does not match claimed expense |
| Amount-cap violation | C. Policy Compliance | Claimed amount exceeds policy cap |
| Invoice timing violation | C. Policy Compliance | Invoice date outside valid period |
| Location deviation | C. Policy Compliance | Expense location inconsistent with business activity |
| Other substantive mismatch | C. Policy Compliance | Other mismatches |
| Invoice forgery | D. Authenticity & Legitimacy | Receipt appears fabricated or tampered |
| Illegitimate purpose | D. Authenticity & Legitimacy | Personal use, quota padding, etc. |
| Duplicate reimbursement | D. Authenticity & Legitimacy | Same expense claimed more than once |
| Required-amount miscalculation | D. Authenticity & Legitimacy | Arithmetic error in claimed amount |
| Irrelevant context | Distractor | Unrelated information added to background |
| Blurred (but valid) invoice image | Distractor | Blurry image that is still a valid receipt |

---

## Inference Script (`infer.py`)

### Overview

`infer.py` reads all workflows, processes source files into a structured prompt adhering to the OpenAI chat message format, and calls an LLM to generate audit responses.

### Key Functions

#### `process_source_files(work_dir, source_files, image_upload_method)`
Reads background documents and receipts from disk and converts them into ordered content blocks (text + images) suitable for multimodal LLMs.

**Supported file types:**
- Images → base64-encoded data URIs
- PDFs → per-page extracted text
- Excel → Markdown tables
- Word → Markdown text (including tables)

#### `build_messages(context, content_blocks, required_amount, currency, expert_prompt)`
Constructs system and user messages for the model. Two system prompt variants are available:
- **`expert_prompt=False`** → Basic system message (`SYSTEM_MSG_NO_CONTEXT`).
- **`expert_prompt=True`** → Expert system message (`SYSTEM_MSG_WITH_CONTEXT`) with a four-stage audit workflow (Authenticity → Compliance → Supporting Docs → Amount Integrity).

#### `call_model_with_retry(messages, **kwargs)`
**⚠️ PLACEHOLDER** — Must be implemented by the user. Expected to:
- Accept messages in OpenAI chat format.
- Return an object with `.content` (response text), `.reasoning_content` (optional), and `.usage` (token counts).
- Handle retries internally.

### Usage

```bash
# 1. Set your API key
set YOUR_API_KEY=sk-xxxxxxxxxxxxxxxx

# 2. Configure and run
python infer.py
```

In `__main__`, configure:
- `root_dir`: Path to `workflow/` directory.
- `models`: List of model identifiers.
- `MAX_TOKENS`: Maximum output tokens.
- `USE_EXPERT_PROMPT`: Whether to use the expert system prompt.
- `IMAGE_UPLOAD_METHOD`: `"direct"` (preserves original format) or `"pil"` (converts to PNG).

**Output:** A JSON file (e.g., `output/260513224817_results.json`) and an incremental JSONL backup. Each entry contains:
- `workflow_id`, `model`, `prompt_version`
- `generated_answer`: The model's full response.
- `reasoning_content`: Chain-of-thought reasoning (if available).
- `usage`: Token usage statistics.
- `latency_seconds`: Response time.

---

## Evaluation Script (`eval.py`)

### Overview

`eval.py` evaluates model outputs against reference answers in two phases:

1. **Exact comparison**: Extracts the FLAG (0/1) and MAX_LEGAL (number) from the first two lines of the model output, then compares against the reference.
2. **LLM-based logic point evaluation**: Uses a separate judge LLM to determine whether each logic point is satisfied by the model's reasoning.

### Key Functions

#### `extract_flag_and_amount(agent_answer)`
Parses the first two meaningful lines of the model output:
- Line 1: `"0"` or `"1"` → FLAG.
- Line 2: A numeric value → MAX_LEGAL.
Returns `(flag: int | None, amount: Decimal | None)`.

#### `grade_flag(pred_flag, ref_flag_raw)`
Compares predicted flag to reference. Returns correctness and error details.

#### `grade_amount(pred_amount, ref_amount_raw, tolerance)`
Compares predicted amount to reference within an absolute tolerance (default: 0.1). Returns correctness, absolute error, and direction (`over` / `under` / `exact`).

#### `call_llm_judge(prompt, api_key, model, use_thinking, max_retries)`
**⚠️ PLACEHOLDER** — Must be implemented by the user. Expected to evaluate logic points and return:
```json
{
    "point_evaluations": {
        "<point_id>": {
            "satisfied": true,
            "comment": "The agent correctly identified the amount-cap violation."
        }
    }
}
```

#### `build_logic_judge_prompt(agent_answer, logic_points)`
Constructs a prompt instructing the judge LLM to evaluate whether the agent's response addresses each logic point.

### Scoring

- FLAG: Exact match → full score / zero.
- MAX_LEGAL: Within tolerance → full score / zero.
- Logic points: Each has a fixed `max_score` from the reference. The judge decides `satisfied` (true/false). The total score is the sum of all satisfied points' max scores.
- **No score redistribution**: The judge does not modify scores; it only decides pass/fail per point.

### Usage

```bash
# 1. Set your judge API key
set YOUR_JUDGE_API_KEY=sk-xxxxxxxxxxxxxxxx

# 2. Configure target result files in __main__ and run
python eval.py
```

In `__main__`, configure:
- `root_dir`: Path to `workflow/` directory.
- `JUDGE_MODEL`: Model identifier for the judge.
- `target_files`: List of result JSON files to evaluate.

**Output:** A JSON file with `_evaled.json` suffix. Each entry contains:
- `score`: Object with `overall_score`, `max_total_score`, `scoring_points` (detailed per-point results).
- `llm_judge_meta`: Judge LLM metadata (usage, latency).
- `llm_judge_error`: Any judge error.

A summary is printed showing per-model/per-prompt:
- Flag accuracy, Amount exact match rate, Logic point average, Average total score.

---

## Failure Mode Analysis (`eval_failure_mode.py`)

### Overview

This script evaluates model outputs for three cognitive failure modes commonly exhibited by LLMs in audit tasks:

| Mode | Name | Description |
|---|---|---|
| Mode 1 | Textual Consistency Blindness | The model fails to notice surface-level contradictions or ambiguities in the documents. |
| Mode 2 | Conflict Self-Rationalization | The model detects a problem but invents a benign excuse (e.g., "probably a typo") instead of flagging it. |
| Mode 3 | Behavioral Intention Reasoning Defect | The model fails to infer fraudulent intent from suspicious patterns, treating deliberate fraud as simple error. |

### Key Functions

#### `call_llm_judge(prompt, api_key, model, ...)`
**⚠️ PLACEHOLDER** — Must be implemented by the user. Expected to return:
```json
{
    "mode1": {"triggered": false, "reason": "..."},
    "mode2": {"triggered": false, "reason": "..."},
    "mode3": {"triggered": false, "reason": "..."}
}
```

### Usage

```bash
python eval_failure_mode.py
```

Configure similarly to `eval.py`:
- `target_files`: List of result JSON files to analyze.
- Output files use `_mode_eval.json` suffix.

---

## Implementation Guide

### Step 1: Implement Model Inference

Replace the placeholder in `infer.py`:

```python
# in run_tests(), inside the placeholder block
result = call_model_with_retry(messages, max_tokens=max_tokens)
```

Implement `call_model_with_retry()`:

```python
def call_model_with_retry(messages, **kwargs):
    # Example using OpenAI API
    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("YOUR_API_KEY"))
    response = client.chat.completions.create(
        model="your-model",
        messages=messages,
        max_tokens=kwargs.get("max_tokens", 4096),
    )
    # Wrap the response to expose .content, .reasoning_content, .usage
    return response
```

### Step 2: Implement LLM Judge

Replace the placeholder in both `eval.py` and `eval_failure_mode.py`:

```python
def call_llm_judge(prompt, api_key, model, use_thinking=False, max_retries=3):
    # Example using DeepSeek API
    import requests
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 2048,
    }
    resp = requests.post(
        "https://api.deepseek.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json=payload,
    )
    content = resp.json()["choices"][0]["message"]["content"]
    import json
    return json.loads(content), None, resp.json().get("usage")
```

### Step 3: Run

```bash
# Inference
python infer.py
# or for a specific mode:
python -c "from infer import run_tests; run_tests('ReimburseBench_v1/workflow', ['gpt-4o'], 'sk-xxx', 'output/results.json')"

# Evaluation
python eval.py

# Failure mode analysis
python eval_failure_mode.py
```

---

## Prompt Output Format

Models are expected to output responses in the following strict format:

```
1
1500.0
--- Reasoning ---
[Step-by-step reasoning citing policy clauses and arithmetic]
--- Problems (if any) ---
Problem: The hotel receipt exceeds the per-night cap of $150 by $120.
```

- **Line 1**: `0` or `1` (FLAG).
- **Line 2**: MAX_LEGAL numeric value.
- **Line 3**: `--- Reasoning ---`.
- **Following lines**: Detailed reasoning.
- **After reasoning**: `--- Problems (if any) ---`.
- **Problems list**: Each problem on its own line prefixed with `Problem: `.

---

## License

This benchmark is released under the Apache License, Version 2.0.
See the [LICENSE](LICENSE) file for details.

Copyright 2026 The ReimburseBench Authors

---

## Citation

If you use ReimburseBench in your research, please cite:

```bibtex
@misc{reimbursebench2025,
  title={ReimburseBench: Benchmarking LMMs as Full-Workflow Auditors for Financial Internal Control},
  author={Anonymous Authors},
  year={2026},
}
```
