"""
Evaluation script for ReimburseBench.
Extracts FLAG and MAX_LEGAL from model outputs, compares against reference answers,
and uses an LLM judge to evaluate logic points.
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
from decimal import Decimal, getcontext
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv

getcontext().prec = 28

# ============================================================
# Configuration
# ============================================================
JUDGE_API_URL = "https://api.deepseek.com/v1/chat/completions"
JUDGE_MODEL_DEFAULT = "deepseek-v4-pro"
JUDGE_USE_THINKING_DEFAULT = False

AMOUNT_ABSOLUTE_TOLERANCE = Decimal("0.1")

# These prefixes match the updated reference_answer.json key names via startswith:
#   "Scoring_point_1_Flag"  starts with "Scoring_point_1"
#   "Scoring_point_2_Max_legal" starts with "Scoring_point_2"
FLAG_POINT_PREFIX = "Scoring_point_1"
AMOUNT_POINT_PREFIX = "Scoring_point_2"

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


def decimal_from(v: Any) -> Optional[Decimal]:
    if v is None:
        return None
    if isinstance(v, Decimal):
        return v
    try:
        return Decimal(str(v))
    except Exception:
        return None


def normalize_yes_no_to_int(raw: Any) -> Optional[int]:
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if s in ("1", "true", "yes", "y", "is"):
        return 1
    if s in ("0", "false", "no", "n", "non"):
        return 0
    m = re.search(r'\b([01])\b', s)
    if m:
        return int(m.group(1))
    return None


def load_reference_points(ref_data: dict) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for k, v in ref_data.items():
        if not k.startswith("Scoring_point_"):
            continue
        if not isinstance(v, dict):
            log(f"  reference field {k} is not a dict, skipping", level="WARN")
            continue
        score = decimal_from(v.get("score"))
        if score is None:
            log(f"  reference field {k} has no valid score, defaulting to 0", level="WARN")
            score = Decimal("0")
        out[k] = {"answer": v.get("answer"), "score": score}
    return out


def find_point_by_prefix(points: Dict[str, Dict[str, Any]],
                         prefix: str) -> Optional[str]:
    candidates = [k for k in points.keys()
                  if k == prefix or k.startswith(prefix + "_")]
    if not candidates:
        return None
    candidates.sort(key=natural_sort_key)
    return candidates[0]


def find_reference_answer(work_dir: str) -> Optional[dict]:
    pattern = os.path.join(work_dir, "*reference_answer*")
    candidates = sorted(glob.glob(pattern))
    if not candidates:
        return None
    if len(candidates) > 1:
        log(f"{work_dir}: found {len(candidates)} reference files, preferring .json",
            level="WARN")
    json_files = [c for c in candidates if c.endswith('.json')]
    chosen = json_files[0] if json_files else candidates[0]
    try:
        with open(chosen, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        log(f"Failed to read reference answer {chosen}: {e}", level="ERROR")
        return None


# ============================================================
# Extract FLAG and MAX_LEGAL from model output
# ============================================================
def extract_flag_and_amount(agent_answer: str) -> Tuple[Optional[int], Optional[Decimal]]:
    """
    Search the full text for the first consecutive two-line pattern:
      Line 1: exactly "0" or "1" (with optional surrounding whitespace)
      Line 2: exactly a number (integer or decimal, with optional surrounding whitespace)
    Returns (int(flag), Decimal(amount)) if found, otherwise (None, None).
    """
    if not agent_answer:
        return None, None

    lines = agent_answer.split('\n')
    for i in range(len(lines) - 1):
        line1 = lines[i].strip()
        line2 = lines[i + 1].strip()

        if line1 not in ('0', '1'):
            continue

        if not re.match(r'^-?\d+(\.\d+)?$', line2):
            continue

        return int(line1), Decimal(line2)

    return None, None


def grade_flag(pred_flag: Optional[int], ref_flag_raw: Any) -> Dict[str, Any]:
    ref_flag = normalize_yes_no_to_int(ref_flag_raw)
    err = None
    if ref_flag is None:
        err = f"reference flag unparseable: {ref_flag_raw!r}"
    elif pred_flag is None:
        err = "no valid 0/1 line found in agent answer"
    correct = (err is None) and (pred_flag == ref_flag)
    return {"pred": pred_flag, "ref": ref_flag, "correct": correct, "error": err}


def grade_amount(pred_amount: Optional[Decimal], ref_amount_raw: Any,
                 tolerance: Decimal = AMOUNT_ABSOLUTE_TOLERANCE) -> Dict[str, Any]:
    ref_amount = decimal_from(ref_amount_raw)
    err = None
    if ref_amount is None:
        err = f"reference amount unparseable: {ref_amount_raw!r}"
    elif pred_amount is None:
        err = "no valid number line found in agent answer"
    if err is not None:
        return {
            "pred": str(pred_amount) if pred_amount is not None else None,
            "ref": str(ref_amount) if ref_amount is not None else None,
            "correct": False, "abs_error": None, "direction": None,
            "tolerance": str(tolerance), "error": err,
        }
    abs_err = abs(pred_amount - ref_amount)
    correct = abs_err < tolerance
    if correct:
        direction = "exact"
    elif pred_amount > ref_amount:
        direction = "over"
    else:
        direction = "under"
    return {
        "pred": str(pred_amount), "ref": str(ref_amount), "correct": correct,
        "abs_error": str(abs_err), "direction": direction,
        "tolerance": str(tolerance), "error": None,
    }


def build_logic_judge_prompt(agent_answer: str,
                             logic_points: Dict[str, Dict[str, Any]]) -> str:
    points_desc_lines = []
    for k in sorted(logic_points.keys(), key=natural_sort_key):
        v = logic_points[k]
        points_desc_lines.append(f'  - Point ID: "{k}"')
        points_desc_lines.append(f'    Max score (already fixed by the rubric, '
                                 f'do not modify): {v["score"]}')
        points_desc_lines.append(f'    Expected answer / content: '
                                 f'"{v.get("answer", "")}"')
    points_desc = "\n".join(points_desc_lines)

    return f"""You are an expert evaluator for a reimbursement audit AI benchmark.

The agent's full answer is given below. The answer follows this structure:
  Line 1: "0" or "1" (a FLAG — whether the application is reimbursable)
  Line 2: a number (the MAX_LEGAL amount)
  Then a reasoning section (starting with "--- Reasoning ---")
  Then a problems section (starting with "--- Problems (if any) ---")

You do NOT evaluate FLAG or MAX_LEGAL — those are scored separately by exact \
comparison against the reference. You ONLY evaluate the listed LOGIC POINTS \
below, which all describe specific problems / audit reasoning that the agent \
should have identified.

For each logic point, decide whether the agent's full answer (especially the \
reasoning and problems sections) substantively addresses or matches the \
expected behaviour. Minor wording differences are fine; semantic content must \
match. The point is satisfied ONLY if the agent's reasoning or problem list \
demonstrably identifies or applies the corresponding audit rule.

Each point already has a fixed max_score (shown below). You MUST NOT \
redistribute, scale, or otherwise modify these scores. You are only deciding \
satisfied = true / false for each point. The script will compute the total \
score by exact arithmetic: total = Σ(satisfied ? max_score : 0). \
Do not output any total / sum yourself.

=== AGENT ANSWER ===
{agent_answer}

=== LOGIC POINTS TO EVALUATE ===
{points_desc}

Output a single JSON object with the following structure (and nothing else):
{{
  "point_evaluations": {{
    "<exact Point ID from above>": {{
      "satisfied": <true or false>,
      "comment": "<one sentence explaining the decision>"
    }}
    // ... one entry per LOGIC POINT listed above
  }}
}}

CRITICAL:
- The keys in "point_evaluations" MUST EXACTLY match the Point IDs above. \
Do not abbreviate, paraphrase, or translate them.
- Include every listed Point ID exactly once.
- Do not invent extra Point IDs.
- Output ONLY the JSON object. No prose before or after, no Markdown fences."""


# ------------------------------------------------------------
# Placeholder for LLM judge API call
# ------------------------------------------------------------
def call_llm_judge(prompt: str, api_key: str, model: str,
                   use_thinking: bool = False,
                   max_retries: int = 3) -> Tuple[Optional[dict], Optional[str], Optional[dict]]:
    """
    PLACEHOLDER: Replace this function with your own LLM judge implementation.

    This function should send the prompt to an LLM that evaluates logic points
    and returns a JSON object with the structure:
      {"point_evaluations": {"<point_id>": {"satisfied": bool, "comment": str}, ...}}

    Expected input:
      - prompt: str (the full evaluation prompt with agent answer and logic points)
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
    points = load_reference_points(ref_data)
    if not points:
        raise JudgeFailure("reference file has no Scoring_point_* entries")

    pred_flag, pred_amount = extract_flag_and_amount(agent_answer)
    flag_key = find_point_by_prefix(points, FLAG_POINT_PREFIX)
    amount_key = find_point_by_prefix(points, AMOUNT_POINT_PREFIX)

    flag_detail = grade_flag(pred_flag, points[flag_key]["answer"]) if flag_key else None
    amount_detail = grade_amount(pred_amount, points[amount_key]["answer"]) if amount_key else None

    logic_points = {k: v for k, v in points.items() if k != flag_key and k != amount_key}

    judge_per_point: Dict[str, Dict[str, Any]] = {}
    judge_meta = None
    judge_error = None
    if logic_points:
        prompt = build_logic_judge_prompt(agent_answer, logic_points)
        parsed, error, meta = call_llm_judge(prompt, api_key, judge_model, use_thinking)
        judge_meta = meta
        if parsed is None:
            raise JudgeFailure(f"LLM judge call failed: {error}")
        judge_error = error
        judge_evals = parsed.get("point_evaluations", {})
        if not isinstance(judge_evals, dict):
            raise JudgeFailure("LLM judge returned point_evaluations that is not a dict")
        for pid in logic_points.keys():
            jr = judge_evals.get(pid)
            if jr is None:
                judge_per_point[pid] = {"satisfied": False,
                                        "comment": "MISSING from judge output",
                                        "missing_from_judge": True}
            else:
                judge_per_point[pid] = {"satisfied": bool(jr.get("satisfied", False)),
                                        "comment": str(jr.get("comment", "")),
                                        "missing_from_judge": False}

    flat: List[Dict[str, Any]] = []
    overall = Decimal("0")
    max_total = Decimal("0")
    for pid in sorted(points.keys(), key=natural_sort_key):
        max_score: Decimal = points[pid]["score"]
        max_total += max_score
        if pid == flag_key:
            correct = bool(flag_detail and flag_detail["correct"])
            detail = flag_detail; kind = "flag"
        elif pid == amount_key:
            correct = bool(amount_detail and amount_detail["correct"])
            detail = amount_detail; kind = "amount"
        else:
            jpp = judge_per_point.get(pid, {})
            correct = bool(jpp.get("satisfied", False))
            detail = jpp; kind = "logic"
        earned = max_score if correct else Decimal("0")
        overall += earned
        flat.append({"id": pid, "kind": kind, "max_score": str(max_score),
                     "correct": correct, "earned": str(earned), "detail": detail})

    return {
        "workflow_id": workflow_id, "model": model_name,
        "prompt_version": prompt_version,
        "overall_score": str(overall), "max_total_score": str(max_total),
        "scoring_points": flat,
        "llm_judge_meta": judge_meta, "llm_judge_error": judge_error,
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


# ============================================================
# Disk write with retry
# ============================================================
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
    log(f"Amount absolute tolerance: {AMOUNT_ABSOLUTE_TOLERANCE}")

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
                    if e.get("score") is not None:
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
                scored = {**entry, "score": None, "scoring_error": reason,
                          "scoring_timestamp": datetime.datetime.now().isoformat()}
                with jsonl_lock:
                    _write_jsonl_line(jsonl_path, scored)
                    counters["errors"] += 1
                log(f"{prefix} {reason}", level=level)

            wf_path = wf_path_map.get(wf_id)
            if not wf_path:
                return fail("workflow folder not found")
            ref_data = find_reference_answer(wf_path)
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
                **entry, "score": score_info,
                "scoring_judge_model": judge_model,
                "scoring_use_thinking": use_thinking,
                "scoring_latency": round(elapsed, 3),
                "scoring_timestamp": datetime.datetime.now().isoformat(),
            }
            with jsonl_lock:
                _write_jsonl_line(jsonl_path, scored)
                counters["done"] += 1
                current_done = counters["done"]

            overall = score_info["overall_score"]
            max_total = score_info["max_total_score"]
            n_pts = len(score_info["scoring_points"])
            n_correct = sum(1 for p in score_info["scoring_points"] if p["correct"])
            log(f"{prefix} -> {overall}/{max_total} | {n_correct}/{n_pts} pts | "
                f"{elapsed:.1f}s")
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
                    scored = {**entry, "score": None,
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
        if e.get("score") is None:
            continue
        groups[(e.get("model", "?"), e.get("prompt_version", "?"))].append(e)

    print("\n" + "=" * 120)
    print(f"{'Model':<28} {'Prompt':<24} {'N':<5} "
          f"{'Flag Acc':<10} {'Amount EM':<11} {'Logic Avg':<11} "
          f"{'Avg Score':<11} {'Max Avg':<10}")
    print("=" * 120)

    for (model, prompt), entries in sorted(groups.items()):
        n = len(entries)
        if n == 0:
            continue
        flag_correct = 0; flag_total = 0
        amount_correct = 0; amount_total = 0
        logic_earned = Decimal("0"); logic_max = Decimal("0")
        overall_sum = Decimal("0"); max_sum = Decimal("0")

        for e in entries:
            sc = e["score"]
            overall_sum += Decimal(sc["overall_score"])
            max_sum += Decimal(sc["max_total_score"])
            for p in sc["scoring_points"]:
                if p["kind"] == "flag":
                    flag_total += 1
                    if p["correct"]: flag_correct += 1
                elif p["kind"] == "amount":
                    amount_total += 1
                    if p["correct"]: amount_correct += 1
                else:
                    logic_max += Decimal(p["max_score"])
                    logic_earned += Decimal(p["earned"])

        flag_acc = (flag_correct / flag_total) if flag_total else float("nan")
        amount_em = (amount_correct / amount_total) if amount_total else float("nan")
        logic_avg = float(logic_earned / logic_max) if logic_max > 0 else float("nan")
        avg_score = float(overall_sum / Decimal(n))
        max_avg = float(max_sum / Decimal(n))
        print(f"{model:<28} {prompt:<24} {n:<5} "
              f"{flag_acc:<10.3f} {amount_em:<11.3f} {logic_avg:<11.3f} "
              f"{avg_score:<11.4f} {max_avg:<10.4f}")
    print("=" * 120)

    print("\n--- Amount direction breakdown ---")
    print(f"{'Model':<28} {'Prompt':<24} {'Exact':<8} {'Over':<8} {'Under':<8} {'N/A':<8}")
    for (model, prompt), entries in sorted(groups.items()):
        dir_counts = defaultdict(int)
        for e in entries:
            amt_pt = next((p for p in e["score"]["scoring_points"]
                           if p["kind"] == "amount"), None)
            d = (amt_pt and (amt_pt.get("detail") or {}).get("direction")) or "n/a"
            dir_counts[d] += 1
        print(f"{model:<28} {prompt:<24} "
              f"{dir_counts['exact']:<8} {dir_counts['over']:<8} "
              f"{dir_counts['under']:<8} {dir_counts['n/a']:<8}")


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
        output_json = file_path[:-5] + '_evaled.json'
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
