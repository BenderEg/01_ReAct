import os, re, json, time, argparse, contextlib
from pathlib import Path
from dataclasses import dataclass, field
import requests
import pandas as pd
import matplotlib.pyplot as plt
from pydantic import BaseModel, Field

from utils import answer_check
from utils.wiki_tools import TOOLS, call_tool

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

assert os.getenv("OPENROUTER_API_KEY"), "нет ключа: положите его в .env рядом с ноутбуком или в Kaggle Secrets"

CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
HEADERS = {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
           "HTTP-Referer": "https://postypashki.ru", "X-Title": "agents-course-seminar01"}
MODELS = {"cheap": "openai/gpt-4o-mini", "mid": "anthropic/claude-haiku-4.5", "strong": "anthropic/claude-sonnet-4.6"}
COLORS = {"violet": "#5436A3", "amber": "#F09000", "teal": "#00838F", "red": "#C43C3C", "grey": "#787882"}
DATA = next((p for p in [Path("qa_data"), Path("/kaggle/input/datasets/artmakar04/"), Path("/kaggle/input/seminar01-data/data")]
             if (p / "compare_10.jsonl").exists()), Path("qa-data"))
TRACES, DATA, IMG, RESULTS = Path("traces"), Path("data"), Path("img"), Path("results")
TRACES.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)
IMG.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)


@dataclass
class Ledger:
    calls: list = field(default_factory=list)

    def add(self, tag, model, usage, seconds=0.0):
        p, c = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
        cost = usage.get("cost") or 0.0
        self.calls.append({"tag": tag, "model": model.split("/")[-1], "prompt": p, "completion": c,
                           "cost": cost, "seconds": round(seconds, 2)})
        return cost

    @property
    def total(self):
        return sum(c["cost"] for c in self.calls)

    def table(self):
        df = pd.DataFrame(self.calls)
        return df.groupby(["tag", "model"]).agg(calls=("cost", "size"), prompt=("prompt", "sum"),
            completion=("completion", "sum"), cost=("cost", "sum"),
            seconds=("seconds", "sum")
        ).round(5)

ledger = Ledger()


class Throttle:
    """Не чаще rpm запросов в минуту к одной модели (новый аккаунт OpenRouter: 20 в минуту на модель)."""

    def __init__(self, rpm: float | None = None) -> None:
        self.rpm = rpm
        self.last: dict[str, float] = {}

    def wait(self, model: str) -> None:
        if not self.rpm:
            return
        gap = 60 / self.rpm - (time.monotonic() - self.last.get(model, float("-inf")))
        if gap > 0:
            time.sleep(gap)
        self.last[model] = time.monotonic()

throttle = Throttle()

RETRY_STATUSES = (429, 500, 502, 503, 504)
MAX_WAIT = 65.0  # окно лимита OpenRouter — минута, плюс запас


def rate_limit_reset(r: requests.Response) -> float | None:
    """Момент снятия лимита (epoch, секунды): заголовок X-RateLimit-Reset или он же в теле 429 от OpenRouter."""
    reset = r.headers.get("X-RateLimit-Reset")
    if reset is None:
        with contextlib.suppress(ValueError, KeyError, TypeError):
            reset = r.json()["error"]["metadata"]["headers"]["X-RateLimit-Reset"]
    with contextlib.suppress(ValueError, TypeError):
        return float(reset) / 1000  # OpenRouter отдаёт миллисекунды
    return None


def retry_delay(r: requests.Response | None, attempt: int) -> float:
    """Сколько ждать перед повтором: Retry-After, затем X-RateLimit-Reset, иначе экспонента."""
    if r is not None:
        retry_after = r.headers.get("Retry-After")
        if retry_after is not None:
            with contextlib.suppress(ValueError):
                return min(MAX_WAIT, max(1.0, float(retry_after)))
        reset = rate_limit_reset(r)
        if reset is not None:
            return min(MAX_WAIT, max(1.0, reset - time.time() + 1))
    return min(60.0, 3.0 * 2 ** attempt)


def post_with_retry(body: dict, attempts: int = 6) -> dict:
    """POST к OpenRouter; при 429/5xx и сетевых ошибках ждёт столько, сколько просит сервер."""
    error = "нет попыток"
    for attempt in range(attempts):
        throttle.wait(body["model"])
        r = None
        try:
            r = requests.post(CHAT_URL, json=body, headers=HEADERS, timeout=120)
        except requests.RequestException as e:
            label, error = type(e).__name__, f"сеть: {type(e).__name__}: {e}"
        else:
            if r.status_code == 200:
                return r.json()
            label, error = f"HTTP {r.status_code}", f"HTTP {r.status_code} : {r.text[:5000]}"
            if r.status_code not in RETRY_STATUSES:
                break
        if attempt < attempts - 1:
            delay = retry_delay(r, attempt)
            print(f"  {label}, жду {delay:.1f} c (попытка {attempt + 2}/{attempts})", flush=True)
            time.sleep(delay)
    raise RuntimeError(error)

def chat(messages: list[dict], model: str, tools: list[dict] | None = None, tag="chat", temperature=None):
    body = {"model": model, "messages": messages, "usage": {"include": True}}
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    if temperature is not None:
        body["temperature"] = temperature
    started = time.perf_counter()
    data = post_with_retry(body)
    ledger.add(tag, model, data.get("usage") or {}, time.perf_counter() - started)
    return data["choices"][0]["message"]

class ExecArgs(BaseModel):
    code: str = Field(description="код на Python; результат надо напечатать через print")

def looped(seen, calls):
    keys = [(c["function"]["name"], c["function"]["arguments"]) for c in calls]
    repeated = any(k in seen for k in keys)
    seen.update(keys)
    return repeated

def finish(model, messages, step):
    del messages[-1]
    messages.append({"role": "user", "content": "Tools are no longer available. Answer from what you already know; "
                                                "write the last line as FINAL: <answer> in English."})
    msg = chat(messages, model, tag="agent")
    messages.append(msg)
    return msg.get("content") or "", step + 1

def agent_loop(messages, model, tool_names, max_steps):
    seen = set()
    for step in range(1, max_steps + 1):
        msg = chat(messages, model, tools=[TOOLS[n]["schema"] for n in tool_names] or None, tag='agent')
        messages.append(msg)
        calls = msg.get("tool_calls") or []

        if not calls:
            return msg.get("content") or "", step
        if looped(seen, calls) or step == max_steps:
            return finish(model, messages, step)

        messages += [run_tool(c) for c in calls]

def run_tool(call):
    name = call["function"]["name"]

    try:
        result = call_tool(name, call["function"]["arguments"])
    except Exception as e:
        result = f"НЕ удалось вызвать инструмент {name}: {e}"
    return {"role": "tool", "tool_call_id" : call["id"], "content": str(result)[:2000]}

def show_trace(run):
    for m in run.messages[1:]:
        calls = "; ".join(f"{c['function']['name']}{c['function']['arguments']}" for c in m.get("tool_calls") or [])
        text = " ".join((m.get("content") or "").split())[:180]
        print(f"{m['role']:9s}| {text} {calls}")
    print(f"шагов: {run.steps}, цена: {run.cost * 100:.3f} ¢, время: {run.seconds:.1f} c")

@dataclass
class Run:
    question: str
    answer: str
    steps: int
    messages: list
    cost: float = 0.0
    seconds: float = 0.0

# Промпт на английском: на русском модель отвечала кириллицей («Криштиану Роналду»),
# а эталоны английские, и верные ответы не засчитывались (см. utils/answer_check.py).
SYSTEM = ("You answer questions about the 2026 FIFA World Cup. "
    "If you need a fact, call the tools instead of guessing. "
    "Mention the 2026 FIFA World Cup in your search queries. "
    "When the answer is ready, write it as the last line in the format FINAL: <answer>. "
    "The FINAL answer must be in English, with names and terms spelled exactly as in the source article; "
    "write numbers as digits; give a short phrase, no explanation. "
    "If you could not find the answer, write FINAL: unknown.")

CONFIGS = {
    "без инструментов": [],
    "поиск": ["web_search"],
    "поиск и чтение страницы": ["web_search", "page_find"],
}


def agent(question, model, tool_names, max_steps=8):
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    before, started = ledger.total, time.perf_counter()
    answer, steps = agent_loop(messages, model, tool_names, max_steps)
    return Run(question, answer, steps, messages, ledger.total - before, time.perf_counter() - started)

ERROR_PREFIX = "ошибка:"

def is_correct(task, answer):
    """Правила сравнения — в utils/answer_check.py; упавший прогон не засчитывается никогда."""
    if answer.startswith(ERROR_PREFIX):
        return False
    return answer_check.is_correct(task["answer"], answer)

def final_answer(text):
    m = re.search(r"FINAL:\s*(.+)", text or "")
    return m.group(1).strip() if m else (text or "").strip()

def run_tasks(tasks, model, tool_names, config, traces: Path = TRACES):
    folder = traces / config / model.split("/")[-1]
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for task in tasks:
        try:
            run = agent(task["question"], model, tool_names)
        except Exception as e:
            run = Run(task["question"], f"{ERROR_PREFIX} {e}", 0, [])
        ok = is_correct(task, final_answer(run.answer))
        rows.append({"config": config, "model": model.split("/")[-1], "id": task["id"], "source": task["source"],
                     "correct": ok, "steps": run.steps, "tool_calls": sum(m["role"] == "tool" for m in run.messages),
                     "cost": run.cost, "seconds": round(run.seconds, 1), "answer": final_answer(run.answer)[:60], "gold": task["answer"]})
        (folder / f"{task['id']}.json").write_text(json.dumps({"task": task, "answer": run.answer, "correct": ok, "cost": run.cost,
                                                              "messages": run.messages}, ensure_ascii=False, indent=1), encoding="utf-8")
    return pd.DataFrame(rows)

def summary(config, model, n, correct, cost, steps, seconds=0.0):
    return {"config": config, "model": model.split("/")[-1], "n": n, "accuracy": round(correct / n, 2),
            "cost_per_task": round(cost / n, 5), "cost_per_correct": round(cost / correct, 5) if correct else float("inf"),
            "avg_steps": round(steps, 2), "avg_seconds": round(seconds, 1)}

def report(results):
    rows = [summary(config, model, len(df), int(df["correct"].sum()), df["cost"].sum(), df["steps"].mean(), df["seconds"].mean())
            for (config, model), df in results.groupby(["config", "model"])]
    return pd.DataFrame(rows).sort_values("cost_per_task").reset_index(drop=True)

QUIZ_PATH = Path("data/fresh.jsonl")

def load_tasks(path: Path = QUIZ_PATH, limit: int | None = None) -> list[dict]:
    """Вопросы из JSONL; run_tasks нужны поля id, source, question, answer."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[:limit] if limit is not None else rows


def config_name(tool_names: list[str]) -> str:
    """Название набора инструментов из CONFIGS, чтобы трейсы и отчёт совпадали с семинаром."""
    for name, tools in CONFIGS.items():
        if tools == tool_names:
            return name
    return "+".join(tool_names) or "без инструментов"


def run_config(level: str, tool_names: list[str], tasks: list[dict] | None = None,
               limit: int | None = None) -> pd.DataFrame:
    """Один запуск: все вопросы через MODELS[level] с инструментами tool_names."""
    model, config = MODELS[level], config_name(tool_names)
    tasks = tasks if tasks is not None else load_tasks(limit=limit)
    print(f"запуск: {config} / {level} ({model}), вопросов: {len(tasks)}", flush=True)
    df = run_tasks(tasks, model, tool_names, config)
    df.to_csv(RESULTS / f"{config}__{level}.csv", index=False, encoding="utf-8")
    print(f"  верно {int(df['correct'].sum())}/{len(df)}, ${df['cost'].sum():.4f}", flush=True)
    return df


def run_experiment(levels: list[str], config_names: list[str], limit: int | None = None) -> pd.DataFrame:
    """Все запуски (набор инструментов x модель) одной таблицей, готовой для report."""
    tasks = load_tasks(limit=limit)
    frames = [run_config(level, CONFIGS[name], tasks=tasks) for name in config_names for level in levels]
    results = pd.concat(frames, ignore_index=True)
    results.to_csv(RESULTS / "all_runs.csv", index=False, encoding="utf-8")
    return results

def rebuild_all_runs() -> pd.DataFrame:
    """Собрать results/all_runs.csv заново из всех results/<конфиг>__<уровень>.csv."""
    results = pd.concat([pd.read_csv(p, encoding="utf-8") for p in sorted(RESULTS.glob("*__*.csv"))],
                        ignore_index=True)
    results.to_csv(RESULTS / "all_runs.csv", index=False, encoding="utf-8")
    return results

def retry_errors(levels: list[str], config_names: list[str]) -> pd.DataFrame:
    """Перезапустить только упавшие задачи (ответ «ошибка: ...») в сохранённых прогонах.

    Строки с ошибкой в results/<конфиг>__<уровень>.csv заменяются новыми, трейсы перезаписывает run_tasks.
    """
    tasks = {t["id"]: t for t in load_tasks()}
    for name in config_names:
        for level in levels:
            csv_path = RESULTS / f"{name}__{level}.csv"
            if not csv_path.exists():
                print(f"{csv_path.name}: нет прогона, пропускаю", flush=True)
                continue
            df = pd.read_csv(csv_path, encoding="utf-8")
            failed = df.loc[df["answer"].astype(str).str.startswith(ERROR_PREFIX), "id"].tolist()
            print(f"{csv_path.name}: перезапуск {len(failed)} упавших", flush=True)
            if not failed:
                continue
            redo = run_tasks([tasks[i] for i in failed], MODELS[level], CONFIGS[name], name)
            order = {task_id: n for n, task_id in enumerate(df["id"])}
            df = (pd.concat([df[~df["id"].isin(failed)], redo], ignore_index=True)
                  .sort_values("id", key=lambda ids: ids.map(order)).reset_index(drop=True))
            df.to_csv(csv_path, index=False, encoding="utf-8")
            print(f"  верно {int(df['correct'].sum())}/{len(df)}, "
                  f"ещё с ошибкой {int(df['answer'].astype(str).str.startswith(ERROR_PREFIX).sum())}", flush=True)
    return rebuild_all_runs()

RERUN = "повторные запуски"

def rerun_failed(levels: list[str], config_names: list[str]) -> pd.DataFrame:
    """Перезапустить все неверные задачи (correct == False) сохранённых прогонов.

    Старые results/<конфиг>__<уровень>.csv и трейсы не трогаются: новые трейсы пишутся в
    traces/повторные запуски/, таблицы — в results/повторные запуски/.
    """
    tasks = {t["id"]: t for t in load_tasks()}
    out = RESULTS / RERUN
    out.mkdir(parents=True, exist_ok=True)
    frames = []
    for name in config_names:
        for level in levels:
            csv_path = RESULTS / f"{name}__{level}.csv"
            if not csv_path.exists():
                print(f"{csv_path.name}: нет прогона, пропускаю", flush=True)
                continue
            df = pd.read_csv(csv_path, encoding="utf-8")
            failed = df.loc[~df["correct"].astype(bool), "id"].tolist()
            print(f"{csv_path.name}: повторный запуск {len(failed)} неверных", flush=True)
            if not failed:
                continue
            redo = run_tasks([tasks[i] for i in failed], MODELS[level], CONFIGS[name], name, traces=TRACES / RERUN)
            redo.to_csv(out / csv_path.name, index=False, encoding="utf-8")
            print(f"  верно {int(redo['correct'].sum())}/{len(redo)}, ${redo['cost'].sum():.4f}, "
                  f"с ошибкой {int(redo['answer'].astype(str).str.startswith(ERROR_PREFIX).sum())}", flush=True)
            frames.append(redo)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def rerun_comparison(levels: list[str], config_names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """До и после правок на одних и тех же задачах плюс сводная точность на всём датасете.

    Только из сохранённых CSV, без вызовов модели. Первая таблица — строки summary для
    старого прогона на перезапущенных id и для повторного; вторая — старые верные плюс
    исправленные повтором на всех вопросах.
    """
    rows, merged = [], []
    for name in config_names:
        for level in levels:
            redo_path = RESULTS / RERUN / f"{name}__{level}.csv"
            if not redo_path.exists():
                continue
            old = pd.read_csv(RESULTS / f"{name}__{level}.csv", encoding="utf-8")
            redo = pd.read_csv(redo_path, encoding="utf-8")
            before = old[old["id"].isin(redo["id"])]
            model = MODELS[level]
            for label, df in (("до правок", before), ("после правок", redo)):
                rows.append(summary(f"{name}, {label}", model, len(df), int(df["correct"].sum()),
                                    df["cost"].sum(), df["steps"].mean(), df["seconds"].mean()))
            kept, fixed = int(old["correct"].sum()), int(redo["correct"].sum())
            merged.append({"config": name, "model": model.split("/")[-1], "n": len(old), "old_correct": kept,
                           "fixed": fixed, "accuracy_before": round(kept / len(old), 2),
                           "accuracy_merged": round((kept + fixed) / len(old), 2)})
    return pd.DataFrame(rows), pd.DataFrame(merged)

def rerun_report(levels: list[str], config_names: list[str], path: str = "img/money_quality_rerun.png") -> None:
    table, merged = rerun_comparison(levels, config_names)
    print(table.to_string(index=False))
    print(merged.to_string(index=False))
    money_chart(table[table["config"].str.endswith("после правок")], path=path)

def rescore() -> pd.DataFrame:
    """Переоценить сохранённые прогоны текущим is_correct без вызовов модели.

    Берёт каждую results/<конфиг>__<уровень>.csv, для каждой строки читает полный ответ из
    трейса, пишет новый correct и в CSV, и в трейс, пересобирает results/all_runs.csv.
    """
    for csv_path in sorted(RESULTS.glob("*__*.csv")):
        df = pd.read_csv(csv_path, encoding="utf-8")
        for idx, row in df.iterrows():
            trace_path = TRACES / row["config"] / row["model"] / f"{row['id']}.json"
            if not trace_path.exists():
                print(f"  нет трейса {trace_path}, оценка оставлена прежней", flush=True)
                continue
            trace = json.loads(trace_path.read_text(encoding="utf-8"))
            ok = is_correct(trace["task"], final_answer(trace["answer"]))
            trace["correct"] = ok
            trace_path.write_text(json.dumps(trace, ensure_ascii=False, indent=1), encoding="utf-8")
            df.at[idx, "correct"] = ok
        df.to_csv(csv_path, index=False, encoding="utf-8")
        print(f"{csv_path.name}: верно {int(df['correct'].sum())}/{len(df)}", flush=True)
    return rebuild_all_runs()

def money_chart(table, path="img/money_quality.png"):
    fig, ax = plt.subplots(figsize=(8, 5))
    for _, r in table.iterrows():
        color = COLORS["red"] if "каскад" in r["config"] else COLORS["amber"] if "sonnet" in r["model"] else COLORS["violet"]
        ax.scatter(r["cost_per_task"] * 100, r["accuracy"] * 100, s=110, color=color)
        ax.annotate(f"{r['config']}\n{r['model']}", (r["cost_per_task"] * 100, r["accuracy"] * 100),
                    fontsize=8, xytext=(6, 4), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("цена задачи, центы (логарифмическая шкала)")
    ax.set_ylabel("доля верных ответов, %")
    ax.grid(alpha=0.3)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="прогнать вопросы из fresh.jsonl по наборам инструментов")
    parser.add_argument("--levels", nargs="+", default=["cheap"], choices=list(MODELS))
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    parser.add_argument("--limit", type=int, default=None, help="взять только первые N вопросов")
    parser.add_argument("--rescore", action="store_true",
                        help="не звать модели, а переоценить сохранённые трейсы текущим is_correct")
    parser.add_argument("--retry-errors", action="store_true",
                        help="перезапустить только задачи, упавшие с ошибкой (например HTTP 429), в сохранённых прогонах")
    parser.add_argument("--rpm", type=float, default=None,
                        help="не больше N запросов в минуту к одной модели (новый аккаунт OpenRouter: 20)")
    parser.add_argument("--rerun-failed", action="store_true",
                        help="перезапустить все неверные задачи; трейсы и CSV — в папки «повторные запуски»")
    parser.add_argument("--rerun-report", action="store_true",
                        help="не звать модели, а пересобрать сравнение до/после и график повторных запусков")
    args = parser.parse_args()
    throttle.rpm = args.rpm

    if args.rescore:
        print(report(rescore()).to_string(index=False))
        raise SystemExit(0)

    if args.rerun_failed or args.rerun_report:
        if args.rerun_failed:
            rerun_failed(args.levels, args.configs)
            print(f"итого потрачено ${ledger.total:.4f}")
        rerun_report(args.levels, args.configs)
        raise SystemExit(0)

    results = (retry_errors(args.levels, args.configs) if args.retry_errors
               else run_experiment(args.levels, args.configs, limit=args.limit))
    table = report(results)
    print(table.to_string(index=False))

    print(f"итого потрачено ${ledger.total:.4f}")
    print(ledger.table().to_string())