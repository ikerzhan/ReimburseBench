"""
Evaluation script for cognitive failure mode analysis.
Evaluates model outputs against three cognitive failure modes:
  - Mode 1: Textual Consistency Blindness
  - Mode 2: Conflict Self-Rationalization
  - Mode 3: Behavioral Intention Reasoning Defect
Uses an LLM judge to determine whether each mode is triggered.
"""

import sys
import os
import re
import json
import glob
import time
import datetime
import random
import threading
import concurrent.futures
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv

# ============================================================
# Configuration
# ============================================================
JUDGE_API_URL = "https://api.deepseek.com/v1/chat/completions"
JUDGE_MODEL_DEFAULT = "deepseek-v4-pro"
JUDGE_USE_THINKING_DEFAULT = False

PROGRESS_LOG_INTERVAL = 10


# ============================================================
# Utilities
# ============================================================
def natural_sort_key(s: str):
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r'(\d+)', s)]


def log(msg: str, level: str = "INFO"):
    ts = datetime.datetime.now().strftime('%H:%M:%S')
    print(f"[{ts}] [{level}] {msg}", flush=True)


def load_reference_data(work_dir: str) -> Optional[dict]:
    """Load the reference answer file from the workflow directory."""
    pattern = os.path.join(work_dir, "*reference_answer*")
    candidates = sorted(glob.glob(pattern))
    if not candidates:
        return None
    json_files = [c for c in candidates if c.endswith('.json')]
    chosen = json_files[0] if json_files else candidates[0]
    try:
        with open(chosen, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        log(f"Failed to read reference answer {chosen}: {e}", level="ERROR")
        return None


def build_judge_prompt(agent_answer: str, reference_answer: dict) -> str:
    """Build the prompt for the LLM judge to evaluate the three cognitive failure modes."""
    ref_text = json.dumps(reference_answer, ensure_ascii=False, indent=2)

    prompt = f"""You are an expert evaluator analyzing an AI agent's output for a reimbursement audit task.

Your task is to determine whether the agent's response exhibits any of the three cognitive failure modes defined below. These modes describe systematic weaknesses of large language models when performing audit-like judgments. You will be given:
1. The agent's full answer.
2. The correct reference answer / ground truth for the task.

Read both carefully. Then, for each of the three modes, decide whether the agent's answer **exhibits the described failure**. Output a single JSON object with your decisions and a short explanation for each mode.

=== DEFINITIONS OF THE THREE FAILURE MODES ===

**Mode 1: Textual Consistency Blindness**
The agent fails to detect literal contradictions, ambiguities, or semantic conflicts within the provided documents (e.g., invoice, application form). The input contains information that is internally inconsistent or ambiguous on its face, but the agent does NOT notice or mention it at all. It proceeds as if everything is perfectly clear.
- Example: An invoice shows a date as "09-04-2022" while the application says "September"; the country uses DD/MM/YY format, so it is actually April. The agent does not flag this ambiguity and just accepts it as September.
- Example: A freight charge appears both in the detail lines and again after the subtotal, causing double counting. The agent does not notice the duplication.
- Key criterion: The agent shows NO awareness of the surface-level anomaly. It does NOT mention any confusion or ask for clarification.

**Mode 2: Conflict Self-Rationalization**
The agent DOES detect a potential issue or inconsistency, but instead of suspending judgment and requesting human clarification, it invents a benign explanation (e.g., "probably a typo") or lowers its own standard (e.g., "might be barely acceptable") to make the problem disappear. The agent acknowledges the anomaly but resolves it unilaterally.
- Example: The agent notices that "frozen fishing bait" is listed under "snacks", thinks "this seems a bit off, but maybe it's acceptable", and approves it anyway.
- Example: The agent sees that the arrival city on a train ticket does not match the application, says "this is likely a typo", and continues without rejecting.
- Key criterion: The agent explicitly mentions a concern or inconsistency, then rationalizes it away without seeking external verification.

**Mode 3: Behavioral Intention Reasoning Defect**
The agent reads the documents and may verify their internal consistency, but it fails to simulate the mental state of the person submitting the expense. It cannot infer whether the described behavior is reasonable or potentially fraudulent according to social norms and business practices. Even if the documents are internally consistent, the overall pattern of activity is suspicious, but the agent does not recognize the suspicious intent.
- Example: A "mop and bucket" appear in a list of high-end gifts for a client. The agent does not question why cleaning tools would be a gift, missing the likely personal misuse.
- Example: Two identical invoices are submitted, but one has been edited (blurred, recolored) to change the invoice number. The agent only notes "duplicate submission", without realizing this is deliberate fraud.
- Key criterion: The behavior pattern suggests an ulterior motive, but the agent treats it as a simple error or acceptable activity, because it cannot reason about the submitter's intentions.

=== AGENT'S ANSWER ===
{agent_answer}

=== REFERENCE ANSWER (GROUND TRUTH) ===
{ref_text}

=== YOUR TASK ===
For each of the three modes, determine if the agent's answer exhibits the failure. Consider both the agent's answer and the reference answer. The reference answer indicates what a correct audit judgment would look like. If the agent's answer contradicts the reference in a way that aligns with a mode's description, mark it as triggered.

Output a single JSON object with exactly this structure (no extra text, no Markdown fences):

{{
  "mode1": {{
    "triggered": true or false,
    "reason": "Brief explanation in English."
  }},
  "mode2": {{
    "triggered": true or false,
    "reason": "Brief explanation in English."
  }},
  "mode3": {{
    "triggered": true or false,
    "reason": "Brief explanation in English."
  }}
}}

CRITICAL:
- Output ONLY the JSON object. No other text before or after.
- The keys must be exactly "mode1", "mode2", "mode3".
- Each "triggered" must be a boolean (true/false).
- Provide a concise but specific reason for each decision.
"""
    return prompt


# ------------------------------------------------------------
# Placeholder for LLM judge API call
# ------------------------------------------------------------
def call_llm_judge(prompt: str, api_key: str, model: str,
                   use_thinking: bool = False,
                   max_retries: int = 3) -> Tuple[Optional[dict], Optional[str], Optional[dict]]:
    """
    PLACEHOLDER: Replace this function with your own LLM judge implementation.

    This function should send the prompt to an LLM that evaluates the three
    cognitive failure modes and returns a JSON object with the structure:
      {"mode1": {"triggered": bool, "reason": str}, ...}

    Expected input:
      - prompt: str (the full evaluation prompt with agent answer and reference)
      - api_key: str (API key for the judge model)
      - model: str (model identifier)
      - use_thinking: bool (whether to enable chain-of-thought)
      - max_retries: int (number of retries on failure)

    Expected output:
      - parsed: Optional[dict] (parsed JSON response)
      - error: Optional[str] (error message if failed)
      - meta: Optional[dict] (metadata like usage, latency, etc.)

    This function should handle retries and error handling internally.
    """
    raise NotImplementedError(
        "LLM judge is not implemented. "
        "Please implement call_llm_judge() to integrate with your judge model API."
    )


class JudgeFailure(Exception):
    pass


def score_one_entry(agent_answer: str, ref_data: dict,
                    api_key: str, judge_model: str,
                    use_thinking: bool = False, *,
                    model_name: str = "", prompt_version: str = "",
                    workflow_id: str = "") -> dict:
    """Evaluate one model response against the three failure modes."""
    prompt = build_judge_prompt(agent_answer, ref_data)
    parsed, error, meta = call_llm_judge(prompt, api_key, judge_model, use_thinking)

    if parsed is None:
        raise JudgeFailure(f"LLM judge call failed: {error}")

    mode_results = {}
    for mode_key in ["mode1", "mode2", "mode3"]:
        mode_data = parsed.get(mode_key, {})
        if not isinstance(mode_data, dict):
            mode_data = {"triggered": False, "reason": "Invalid judge output"}
        mode_results[mode_key] = {
            "triggered": bool(mode_data.get("triggered", False)),
            "reason": str(mode_data.get("reason", ""))
        }

    return {
        "workflow_id": workflow_id,
        "model": model_name,
        "prompt_version": prompt_version,
        "mode_analysis": mode_results,
        "llm_judge_meta": meta,
        "llm_judge_error": error,
    }


def build_wf_path_map(root_dir: str) -> Dict[str, str]:
    wf_path_map: Dict[str, str] = {}
    duplicates: List[Tuple[str, str, str]] = []
    for sub in sorted(os.listdir(root_dir)):
        sub_path = os.path.join(root_dir, sub)
        if not os.path.isdir(sub_path):
            continue
        task_files = glob.glob(os.path.join(sub_path, "*task_description*"))
        if task_files:
            try:
                with open(task_files[0], 'r', encoding='utf-8') as f:
                    task_data = json.load(f)
                wf_id = task_data.get("id", sub)
            except Exception as e:
                log(f"Failed to read {task_files[0]}: {e}", level="WARN")
                wf_id = sub
        else:
            wf_id = sub
        if wf_id in wf_path_map:
            duplicates.append((wf_id, wf_path_map[wf_id], sub_path))
        wf_path_map[wf_id] = sub_path
    if duplicates:
        log(f"Detected duplicate workflow_ids ({len(duplicates)} occurrences):", level="WARN")
        for wf_id, p1, p2 in duplicates:
            log(f"  - {wf_id}: {p1} <- {p2}", level="WARN")
    return wf_path_map


def _write_jsonl_line(jsonl_path: str, entry: dict, max_retries: int = 5):
    last_err = None
    for attempt in range(max_retries):
        try:
            with open(jsonl_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            return
        except (PermissionError, OSError) as e:
            last_err = e
            sleep_s = 0.1 * (2 ** attempt) + random.uniform(0, 0.1)
            log(f"  Transient write error (attempt {attempt+1}/{max_retries}): {e}; "
                f"waiting {sleep_s:.2f}s to retry", level="WARN")
            time.sleep(sleep_s)
    raise last_err


def batch_scoring(results_json_path: str, root_dir: str, api_key: str,
                  output_score_path: str,
                  judge_model: str = JUDGE_MODEL_DEFAULT,
                  use_thinking: bool = JUDGE_USE_THINKING_DEFAULT,
                  num_workers: int = 10) -> List[dict]:
    log(f"Reading generation results: {results_json_path}")
    with open(results_json_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    log(f"Total {len(results)} entries to score")
    log(f"Judge model: {judge_model}, thinking: {use_thinking}")

    log(f"Scanning workflow directory: {root_dir}")
    wf_path_map = build_wf_path_map(root_dir)
    log(f"Loaded {len(wf_path_map)} workflow paths")

    jsonl_path = output_score_path.replace('.json', '.jsonl')

    already_scored = set()
    if os.path.exists(jsonl_path):
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    e = json.loads(line)
                    if e.get("mode_analysis") is not None:
                        key = (e.get("workflow_id"), e.get("model"),
                               e.get("prompt_version", ""))
                        already_scored.add(key)
                except Exception:
                    pass
        log(f"Found {len(already_scored)} already-scored entries, will skip")

    pending = []
    for idx, entry in enumerate(results, 1):
        wf_id = entry.get("workflow_id")
        model_name = entry.get("model", "unknown")
        prompt_ver = entry.get("prompt_version", "")
        if (wf_id, model_name, prompt_ver) in already_scored:
            continue
        pending.append((idx, entry))

    log(f"Entries requiring scoring: {len(pending)}")

    if pending:
        t_all_start = time.time()
        counters = {"done": 0, "errors": 0, "missing": 0}
        jsonl_lock = threading.Lock()

        def _process_entry_inner(idx, entry, prefix):
            wf_id = entry.get("workflow_id")
            model_name = entry.get("model", "unknown")
            prompt_ver = entry.get("prompt_version", "")
            agent_answer = entry.get("generated_answer", "")

            def fail(reason: str, level: str = "WARN"):
                scored = {**entry, "mode_analysis": None, "scoring_error": reason,
                          "scoring_timestamp": datetime.datetime.now().isoformat()}
                with jsonl_lock:
                    _write_jsonl_line(jsonl_path, scored)
                    counters["errors"] += 1
                log(f"{prefix} {reason}", level=level)

            wf_path = wf_path_map.get(wf_id)
            if not wf_path:
                return fail("workflow folder not found")
            ref_data = load_reference_data(wf_path)
            if not ref_data:
                return fail("reference answer missing")

            t_start = time.time()
            try:
                score_info = score_one_entry(
                    agent_answer, ref_data, api_key, judge_model, use_thinking,
                    model_name=model_name, prompt_version=prompt_ver,
                    workflow_id=wf_id,
                )
            except JudgeFailure as e:
                return fail(f"judge failure: {e}", level="ERROR")
            except Exception as e:
                log(f"{prefix} scoring exception: {e}", level="ERROR")
                import traceback; traceback.print_exc()
                return fail(f"scoring exception: {e}", level="ERROR")

            elapsed = time.time() - t_start
            scored = {
                **entry, "mode_analysis": score_info["mode_analysis"],
                "scoring_judge_model": judge_model,
                "scoring_use_thinking": use_thinking,
                "scoring_latency": round(elapsed, 3),
                "scoring_timestamp": datetime.datetime.now().isoformat(),
            }
            with jsonl_lock:
                _write_jsonl_line(jsonl_path, scored)
                counters["done"] += 1
                current_done = counters["done"]

            triggered = [k for k, v in score_info["mode_analysis"].items() if v["triggered"]]
            log(f"{prefix} -> triggered modes: {triggered if triggered else 'none'} | {elapsed:.1f}s")
            if current_done % PROGRESS_LOG_INTERVAL == 0:
                log(f"Progress: {current_done}/{len(pending)} completed")

        def process_entry(idx, entry):
            wf_id = entry.get("workflow_id")
            model_name = entry.get("model", "unknown")
            prompt_ver = entry.get("prompt_version", "")
            prefix = f"[{idx}/{len(results)}] {wf_id} [{model_name}/{prompt_ver}]"

            try:
                _process_entry_inner(idx, entry, prefix)
            except BaseException as e:
                import traceback
                tb = traceback.format_exc()
                log(f"{prefix} top-level exception (fallback): {e}\n{tb}", level="ERROR")
                try:
                    scored = {**entry, "mode_analysis": None,
                              "scoring_error": f"top-level exception: {e}",
                              "scoring_timestamp": datetime.datetime.now().isoformat()}
                    with jsonl_lock:
                        _write_jsonl_line(jsonl_path, scored)
                        counters["missing"] += 1
                except BaseException as e2:
                    with jsonl_lock:
                        counters["missing"] += 1
                    log(f"{prefix} fallback write also failed: {e2}", level="ERROR")

        log(f"Starting multi-threaded evaluation, concurrency: {num_workers}")
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_entry, idx, entry)
                       for idx, entry in pending]
            for fut in concurrent.futures.as_completed(futures):
                try:
                    fut.result()
                except BaseException as e:
                    log(f"future exception not caught (should not happen): {e}", level="ERROR")

        elapsed_all = time.time() - t_all_start
        total_handled = counters["done"] + counters["errors"] + counters["missing"]
        log(f"\nScoring complete: success {counters['done']}, errors {counters['errors']}, "
            f"fallback {counters['missing']}, total {total_handled}/{len(pending)} "
            f"(elapsed {elapsed_all:.1f}s)")
        if total_handled != len(pending):
            log(f"Mismatch! Missing {len(pending) - total_handled} entries, "
                f"check logs for 'future exception' lines", level="ERROR")

    log("Merging JSONL to final JSON...")
    all_entries: List[dict] = []
    if os.path.exists(jsonl_path):
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    all_entries.append(json.loads(line))
                except Exception:
                    pass
    with open(output_score_path, 'w', encoding='utf-8') as f:
        json.dump(all_entries, f, indent=2, ensure_ascii=False)
    log(f"Scoring results saved to: {output_score_path}")
    log(f"JSONL backup at: {jsonl_path}")

    print_summary(all_entries)
    return all_entries


def print_summary(scored_results: List[dict]):
    groups = defaultdict(list)
    for e in scored_results:
        if e.get("mode_analysis") is None:
            continue
        groups[(e.get("model", "?"), e.get("prompt_version", "?"))].append(e)

    print("\n" + "=" * 100)
    print(f"{'Model':<28} {'Prompt':<24} {'N':<5} "
          f"{'Mode1 Rate':<12} {'Mode2 Rate':<12} {'Mode3 Rate':<12} "
          f"{'Any Mode':<10}")
    print("=" * 100)

    for (model, prompt), entries in sorted(groups.items()):
        n = len(entries)
        if n == 0:
            continue
        m1 = sum(1 for e in entries if e["mode_analysis"]["mode1"]["triggered"])
        m2 = sum(1 for e in entries if e["mode_analysis"]["mode2"]["triggered"])
        m3 = sum(1 for e in entries if e["mode_analysis"]["mode3"]["triggered"])
        any_mode = sum(1 for e in entries if any(e["mode_analysis"][k]["triggered"] for k in ["mode1","mode2","mode3"]))
        print(f"{model:<28} {prompt:<24} {n:<5} "
              f"{m1/n:<12.3f} {m2/n:<12.3f} {m3/n:<12.3f} "
              f"{any_mode/n:<10.3f}")
    print("=" * 100)


# ============================================================
# Entry point
# ============================================================
if __name__ == "__main__":
    load_dotenv()
    root_dir = "ReimburseBench_v1/workflow"
    api_key = os.environ.get("YOUR_JUDGE_API_KEY")
    if not api_key:
        log("Please set the environment variable YOUR_JUDGE_API_KEY", level="ERROR")
        sys.exit(1)

    JUDGE_MODEL = "your-judge-model"
    USE_THINKING = False

    # Example: list paths to model output JSON files
    target_files = [
        r'output/example_results.json',
    ]

    log(f"Found {len(target_files)} files to evaluate.")
    global_all_entries = []

    for file_path in target_files:
        output_json = file_path[:-5] + '_mode_eval.json'
        log("=" * 60)
        log(f"Starting evaluation: {file_path}")
        log(f"Output path: {output_json}")
        log("=" * 60)
        entries = batch_scoring(
            results_json_path=file_path, root_dir=root_dir, api_key=api_key,
            output_score_path=output_json, judge_model=JUDGE_MODEL,
            use_thinking=USE_THINKING, num_workers=10,
        )
        if entries:
            global_all_entries.extend(entries)

    log("\nAll files evaluated!")
    if len(target_files) > 1 and global_all_entries:
        print("\n" + "=" * 45 + " GLOBAL SUMMARY " + "=" * 45)
        print_summary(global_all_entries)
