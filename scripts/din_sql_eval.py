"""DIN-SQL prompt cascade on top of a trained Spider LoRA.

Implements the Pourreza & Rafiei (NeurIPS 2023) DIN-SQL pipeline:
    1. Schema linking
    2. Difficulty classification
    3. Category-specific SQL generation
    4. Self-correction
    5. Execution verification (our addition — retry on SQLite error)

Each stage is a separate vllm batch call across all questions, so we
get N-way batching efficiency at every stage. Total inference cost is
~3-5× a single SQL agent rollout but yields the +20-30pp gain DIN-SQL
reports on top of GPT-4 (which was 75% raw → 85% with cascade).

Usage:
    python scripts/din_sql_eval.py \
        --base /root/autodl-tmp/models/Qwen3-4B-Instruct-2507 \
        --lora /root/autodl-tmp/lora_only_v2/step_200 \
        --spider-data /root/autodl-tmp/datasets/spider/spider_data \
        --split validation --n 1034 \
        --output /root/autodl-tmp/v2_dinsql_results.json
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from agent_system.environments.env_package.spider.sql_env import (
    SpiderSQLEnv, _exec_match, _is_order_sensitive, _normalise_rows,
)


# ──────────────────────────────────────────────────────────────────
# Prompt templates (Stage 1-4)
# ──────────────────────────────────────────────────────────────────

SCHEMA_LINKING_PROMPT = """You are an expert SQL analyst. Given a database schema and a natural language question, identify exactly which tables, columns, and joins are needed.

Schema:
{schema}

Question: {question}

Output a JSON object with these keys:
- "tables": list of table names that must be referenced
- "columns": list of "table.column" strings that the question asks about (in SELECT or WHERE)
- "joins": list of join conditions like "table_a.col = table_b.col" if multiple tables involved
- "conditions": list of WHERE conditions implied by the question
- "groupby": list of columns to GROUP BY if aggregation needed (else empty)
- "orderby": list of {{"col": "name", "direction": "ASC|DESC"}} if ordering needed (else empty)

Output only the JSON, nothing else.

JSON:"""


CLASSIFICATION_PROMPT = """Classify the SQL difficulty for this question.

Schema (compact):
{compact_schema}

Question: {question}
Schema linking: {linking}

Categories:
- "easy"        : single table, simple WHERE/SELECT, maybe COUNT, no GROUP BY, no JOIN
- "non-nested"  : involves JOIN of 2+ tables OR GROUP BY OR aggregation, but NO subqueries
- "nested"      : requires subquery (nested SELECT, EXCEPT, INTERSECT, UNION, or correlated subquery)

Output exactly one word: easy / non-nested / nested

Category:"""


GEN_EASY_PROMPT = """Generate a SQL query for this single-table or simple multi-condition question.

Schema:
{schema}

Question: {question}
Schema linking: {linking}

Examples:
Q: How many singers do we have?
A: SELECT COUNT(*) FROM singer

Q: Names of all singers from France
A: SELECT name FROM singer WHERE country = 'France'

Q: Maximum age of singer
A: SELECT MAX(age) FROM singer

Output the SQL. ASCII only. No comments. No markdown fences. Single statement.

SQL:"""


GEN_NON_NESTED_PROMPT = """Generate a SQL query that joins tables and/or uses GROUP BY/aggregation. Do NOT use subqueries.

Schema:
{schema}

Question: {question}
Schema linking: {linking}

Examples:
Q: Names and country of singers who have at least 2 songs
A: SELECT s.name, s.country FROM singer s JOIN song g ON g.singer_id = s.id GROUP BY s.id HAVING COUNT(g.id) >= 2

Q: Total budget per department
A: SELECT department_id, SUM(budget) FROM project GROUP BY department_id

Q: Singers and concert count, ordered by count desc
A: SELECT s.name, COUNT(c.id) FROM singer s LEFT JOIN concert c ON c.singer_id = s.id GROUP BY s.id ORDER BY COUNT(c.id) DESC

Output the SQL. ASCII only. No comments. No markdown fences. Single statement.

SQL:"""


GEN_NESTED_PROMPT = """Generate a SQL query that requires a subquery (nested SELECT, EXCEPT, INTERSECT, or UNION).

Schema:
{schema}

Question: {question}
Schema linking: {linking}

Approach:
1. Inner query: compute the subset / aggregate first
2. Outer query: use the inner result for filter / comparison

Examples:
Q: Singer with most concerts
Inner: SELECT singer_id, COUNT(*) c FROM concert GROUP BY singer_id ORDER BY c DESC LIMIT 1
Outer: SELECT name FROM singer WHERE id = <inner.singer_id>
A: SELECT name FROM singer WHERE id = (SELECT singer_id FROM concert GROUP BY singer_id ORDER BY COUNT(*) DESC LIMIT 1)

Q: Departments not having any project
Inner: SELECT department_id FROM project
A: SELECT name FROM department WHERE id NOT IN (SELECT department_id FROM project)

Q: Singers older than the average age
Inner: SELECT AVG(age) FROM singer
A: SELECT name FROM singer WHERE age > (SELECT AVG(age) FROM singer)

Output the SQL. ASCII only. No comments. No markdown fences. Single statement.

SQL:"""


CORRECTION_PROMPT = """Check this SQL query for correctness against the schema. Fix any issues.

Schema:
{schema}

Question: {question}
Generated SQL: {sql}
{error_section}

Common issues to check:
- Column or table name typos (must match schema exactly, case-sensitive)
- Missing JOIN when columns are from multiple tables
- GROUP BY missing required columns
- Aggregation in WHERE (use HAVING instead)
- ORDER BY ASC vs DESC for "most/least", "first/last", etc.
- Quotes around string literals
- Subquery returning multiple rows but used with =

If the SQL looks correct, output it unchanged. If incorrect, output the fixed SQL.
ASCII only. No comments. No markdown fences. Single statement.

SQL:"""


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────

def compact_schema(full_schema: str, max_chars: int = 1500) -> str:
    """Schema linking gets the full schema, but classifier and gen prompts
    use a more compact version when the original is huge."""
    if len(full_schema) <= max_chars:
        return full_schema
    # Keep the table headers and first 2 columns of each
    lines = full_schema.split("\n")
    out = []
    skip_block = False
    line_count_in_table = 0
    for line in lines:
        if line.startswith("CREATE TABLE"):
            out.append(line)
            line_count_in_table = 0
            skip_block = False
        elif "PRIMARY KEY" in line or "FOREIGN KEY" in line or line == ");":
            out.append(line)
        elif not skip_block:
            line_count_in_table += 1
            if line_count_in_table <= 4:
                out.append(line)
            else:
                if line_count_in_table == 5:
                    out.append("  -- (more columns elided)")
                skip_block = True
    return "\n".join(out)


def parse_json_loose(text: str) -> dict:
    """Best-effort JSON parse — sometimes the model adds prose around it."""
    text = text.strip()
    # Find first { and last }
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return {}
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        # Try to repair: remove trailing commas, etc.
        return {}


def extract_sql(text: str) -> str:
    """Strip any markdown / commentary, get the SQL."""
    text = text.strip()
    # Remove markdown code fences
    if "```sql" in text.lower():
        m = re.search(r"```sql\s*(.+?)```", text, re.DOTALL | re.IGNORECASE)
        if m:
            text = m.group(1)
    elif text.startswith("```"):
        text = text.strip("`").strip()
    # Remove leading "SQL:" prefix
    text = re.sub(r"^SQL:\s*", "", text, flags=re.IGNORECASE)
    # Take only first statement (split on ; outside parens — naive)
    text = text.split(";")[0]
    return text.strip()


def classify_strict(text: str) -> str:
    text = text.strip().lower()
    if "easy" in text:
        return "easy"
    if "nested" in text and "non-nested" not in text and "non nested" not in text:
        return "nested"
    if "non-nested" in text or "non nested" in text:
        return "non-nested"
    return "non-nested"  # default


# ──────────────────────────────────────────────────────────────────
# DIN-SQL pipeline
# ──────────────────────────────────────────────────────────────────

def chat_format(prompt: str, tokenizer) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def run_pipeline(
    questions: list[dict],
    llm,
    tokenizer,
    sampling_params,
    sampling_params_short,
    lora_request,
    spider_data: str,
    split: str,
    max_correction_attempts: int = 2,
):
    """Run the full DIN-SQL cascade on a list of questions.

    Each question dict has keys: idx, db_id, question, schema, gold_sql, db_path
    """
    n = len(questions)
    print(f"[pipeline] {n} questions; max_correction_attempts={max_correction_attempts}")

    # ── Stage 1: Schema linking ─────────────────────────────────
    t0 = time.perf_counter()
    prompts = [
        chat_format(SCHEMA_LINKING_PROMPT.format(schema=q["schema"], question=q["question"]),
                    tokenizer)
        for q in questions
    ]
    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request, use_tqdm=False)
    for q, out in zip(questions, outputs):
        q["linking"] = out.outputs[0].text.strip()
    print(f"  Stage 1 (schema linking): {time.perf_counter() - t0:.1f}s")

    # ── Stage 2: Classification ─────────────────────────────────
    t0 = time.perf_counter()
    prompts = [
        chat_format(
            CLASSIFICATION_PROMPT.format(
                compact_schema=compact_schema(q["schema"]),
                question=q["question"],
                linking=q["linking"][:500],  # truncate
            ),
            tokenizer,
        )
        for q in questions
    ]
    outputs = llm.generate(prompts, sampling_params_short, lora_request=lora_request, use_tqdm=False)
    for q, out in zip(questions, outputs):
        q["category"] = classify_strict(out.outputs[0].text)
    cat_counts = {}
    for q in questions:
        cat_counts[q["category"]] = cat_counts.get(q["category"], 0) + 1
    print(f"  Stage 2 (classify): {time.perf_counter() - t0:.1f}s  dist={cat_counts}")

    # ── Stage 3: SQL Generation ──────────────────────────────────
    t0 = time.perf_counter()
    prompts = []
    for q in questions:
        if q["category"] == "easy":
            tmpl = GEN_EASY_PROMPT
        elif q["category"] == "non-nested":
            tmpl = GEN_NON_NESTED_PROMPT
        else:
            tmpl = GEN_NESTED_PROMPT
        p = tmpl.format(
            schema=q["schema"],
            question=q["question"],
            linking=q["linking"][:500],
        )
        prompts.append(chat_format(p, tokenizer))
    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request, use_tqdm=False)
    for q, out in zip(questions, outputs):
        q["sql_initial"] = extract_sql(out.outputs[0].text)
    print(f"  Stage 3 (generate): {time.perf_counter() - t0:.1f}s")

    # ── Stage 4-5: Self-correction with execution feedback ──────
    # Initial SQL → check exec → if error or correction wanted, re-prompt
    for q in questions:
        q["sql"] = q["sql_initial"]
        q["correction_attempts"] = 0
        q["last_error"] = None

    for attempt in range(max_correction_attempts + 1):  # +1 for initial
        # Test execution on each question
        need_correction = []
        for q in questions:
            if q.get("done"):
                continue
            try:
                conn = sqlite3.connect(str(q["db_path"]))
                conn.text_factory = str
                conn.execute(q["sql"]).fetchmany(1)
                conn.close()
                q["done"] = True
            except sqlite3.Error as e:
                q["last_error"] = str(e)[:300]
                if attempt < max_correction_attempts:
                    need_correction.append(q)
                else:
                    q["done"] = True  # give up after max attempts

        if not need_correction:
            break

        print(f"  Stage 4 attempt {attempt+1}: {len(need_correction)} need correction")
        prompts = []
        for q in need_correction:
            err_section = (
                f"\nExecution error: {q['last_error']}\n" if q['last_error'] else ""
            )
            p = CORRECTION_PROMPT.format(
                schema=q["schema"],
                question=q["question"],
                sql=q["sql"],
                error_section=err_section,
            )
            prompts.append(chat_format(p, tokenizer))
        t0 = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params, lora_request=lora_request, use_tqdm=False)
        for q, out in zip(need_correction, outputs):
            q["sql"] = extract_sql(out.outputs[0].text)
            q["correction_attempts"] += 1
        print(f"    correction took {time.perf_counter() - t0:.1f}s")

    # ── Final: Score against gold ───────────────────────────────
    n_won = 0
    for q in questions:
        match, err = _exec_match(q["sql"], q["gold_sql"], q["db_path"])
        q["won"] = match
        q["final_error"] = err
        if match:
            n_won += 1

    return {
        "n": n,
        "n_won": n_won,
        "pass_at_1": n_won / n if n else 0,
        "category_dist": cat_counts,
        "avg_correction_attempts": sum(q.get("correction_attempts", 0) for q in questions) / n,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--lora", required=True, help="path to LoRA adapter dir")
    parser.add_argument("--spider-data", required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--n", type=int, default=1034)
    parser.add_argument("--max-corrections", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--gpu-mem-util", type=float, default=0.7)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    print(f"[setup] base={args.base}")
    print(f"[setup] lora={args.lora}")
    print(f"[setup] split={args.split} n={args.n}")

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    llm = LLM(
        model=args.base,
        enable_lora=True,
        max_lora_rank=args.lora_rank,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=4096,
        dtype="bfloat16",
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=1.0,
        max_tokens=args.max_tokens,
    )
    sampling_params_short = SamplingParams(
        temperature=args.temperature,
        top_p=1.0,
        max_tokens=64,
    )

    lora_request = LoRARequest(
        lora_name="active",
        lora_int_id=1,
        lora_path=args.lora,
    )

    # Load all questions + their schemas
    print(f"[loading] {args.n} questions from {args.split} ...")
    env = SpiderSQLEnv(data_dir=args.spider_data, split=args.split)
    questions = []
    for i in range(args.n):
        env.reset(i)
        # Use env's loaded data
        ex = env._current
        questions.append({
            "idx": i,
            "db_id": ex["db_id"],
            "question": ex["question"],
            "schema": env._render_schema(ex["db_id"], env._db_path),
            "gold_sql": ex["query"],
            "db_path": env._db_path,
        })
    print(f"[loading] done ({len(questions)} questions)")

    # Run pipeline
    t0 = time.perf_counter()
    result = run_pipeline(
        questions, llm, tokenizer, sampling_params, sampling_params_short,
        lora_request, args.spider_data, args.split, args.max_corrections,
    )
    result["wall_seconds"] = time.perf_counter() - t0
    print(f"\n[result] pass@1 = {result['pass_at_1']:.4f} "
          f"({result['n_won']}/{result['n']})")
    print(f"         category dist: {result['category_dist']}")
    print(f"         avg corrections: {result['avg_correction_attempts']:.2f}")
    print(f"         wall: {result['wall_seconds']:.0f}s")

    # Save full results (including per-question for analysis)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Per-question results
    per_q = [
        {
            "idx": q["idx"],
            "db_id": q["db_id"],
            "question": q["question"],
            "category": q.get("category"),
            "sql_initial": q.get("sql_initial"),
            "sql_final": q.get("sql"),
            "won": q.get("won"),
            "correction_attempts": q.get("correction_attempts", 0),
            "final_error": q.get("final_error", ""),
        }
        for q in questions
    ]
    out_path.write_text(json.dumps({"summary": result, "per_question": per_q},
                                    indent=2, default=str))
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
