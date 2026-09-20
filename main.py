import os, re, json, time, argparse
from pathlib import Path
from dataclasses import dataclass, field
import requests
import pandas as pd
import matplotlib.pyplot as plt
from pydantic import BaseModel, Field

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


def post_with_retry(body, attempts=3):
    for attempt in range(attempts):
        r = requests.post(CHAT_URL, json=body, headers=HEADERS, timeout=120)
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 500, 502, 503) and attempt < attempts - 1:
            time.sleep(1.5 * (attempt+1))
            continue
        raise RuntimeError(f"HTTP {r.status_code} : {r.text[:5000]}")

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

WIKI = "https://en.wikipedia.org/w/api.php"
WIKI_HEADERS = {"User-Agent": "agents-course-seminar01/1.0 (https://postypashki.ru; educational project)"}

def wiki(params: dict, attempts: int = 3):
    for attempt in range(attempts):
        r = requests.get(WIKI, params={**params, "format": "json"}, headers=WIKI_HEADERS, timeout=15)
        if r.status_code == 200 and r.headers.get("content-type", "").startswith("application/json"):
            return r.json()["query"]
        time.sleep(1.0 + attempt)
    raise RuntimeError(f"Википедия ответила {r.status_code}")

class SearchArgs(BaseModel):
    query: str = Field(description="короткий поисковый запрос: имя, название, термин")

class PageFindArgs(BaseModel):
    title: str = Field(description="точное название статьи")
    keywords: str = Field(description="от двух до пяти ключевых слов из вопроса")

class ExecArgs(BaseModel):
    code: str = Field(description="код на Python; результат надо напечатать через print")

def web_search(query):
    hits = wiki({"action": "query", "list": "search", "srsearch": query, "srlimit": 3})["search"]
    if not hits:
        return "nothing found"
    pages = wiki({"action": "query", "prop": "extracts", "explaintext": 1, "exintro": 1, "titles": hits[0]["title"]})["pages"]
    text = " ".join(next(iter(pages.values())).get("extract", "").split())[:1500]
    others = ", ".join(h["title"] for h in hits[1:])
    return f"[{hits[0]['title']}] {text}" + (f" | other articles: {others}" if others else "")

def page_find(title, keywords):
    """Строки полного текста статьи, где встречается хотя бы половина ключевых слов."""
    
    pages = wiki({"action": "query", "prop": "extracts", "explaintext": 1, "titles": title, "redirects": 1})["pages"]
    text = next(iter(pages.values())).get("extract", "")
    if not text:
        return "no such page"
    words = [w for w in re.findall(r"\w+", keywords.lower()) if len(w) > 2]
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    hits = [l for l in lines if sum(w in l.lower() for w in words) >= max(1, (len(words) + 1) // 2)]
    return "\n".join(h[:400] for h in hits[:6]) or "keywords not found on the page"

TOOLS = {
    "web_search": {"fn": web_search, "args": SearchArgs, "schema": {"type": "function", "function": {
        "name": "web_search", "description": "Searches English Wikipedia and returns the intro of the best article and the titles of two more",
        "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "short query: a name, a title, a term"}},
                       "required": ["query"]}}}},
    "page_find": {"fn": page_find, "args": PageFindArgs, "schema": {"type": "function", "function": {
        "name": "page_find", "description": "Looks inside the full text of a Wikipedia article (for example '2026 in Japan') and returns the lines containing the keywords",
        "parameters": {"type": "object", "properties": {"title": {"type": "string", "description": "exact article title"},
                                                        "keywords": {"type": "string", "description": "two to five keywords from the question"}},
                       "required": ["title", "keywords"]}}}},
}

def looped(seen, calls):
    keys = [(c["function"]["name"], c["function"]["arguments"]) for c in calls]
    repeated = any(k in seen for k in keys)
    seen.update(keys)
    return repeated

def finish(model, messages, step):
    del messages[-1]
    messages.append({"role": "user", "content": "Инструменты больше недоступны. Ответь по тому, что уже известно, последней строкой FINAL: <ответ>."})
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
        spec = TOOLS[name]
        args = spec["args"].model_validate_json(call["function"]["arguments"])
        result = spec["fn"](**args.model_dump())
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

SYSTEM = ("Ты отвечаешь на вопросы про чемпионат мира 2026 года."
    "Если нужно найти факт, вызывай инструменты, а не угадывай. "
    "Добавляй в поисковые запросы уточнение, что запрос про чемпионат мира 2026 года"
    "Когда ответ готов, напиши его последней строкой в формате FINAL: <ответ>. "
    "Для числовых задач в FINAL только число, для вопросов о фактах короткая фраза.")

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

def normalize(text):
    text = re.sub(r"[^\w\s]", " ", str(text).lower().replace(",", ""))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())

def is_correct(task, answer):
    return normalize(task["answer"]) in normalize(answer)

def final_answer(text):
    m = re.search(r"FINAL:\s*(.+)", text or "")
    return m.group(1).strip() if m else (text or "").strip()

def run_tasks(tasks, model, tool_names, config):
    folder = TRACES / config / model.split("/")[-1]
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for task in tasks:
        try:
            run = agent(task["question"], model, tool_names)
        except Exception as e:
            run = Run(task["question"], f"ошибка: {e}", 0, [])
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
    args = parser.parse_args()

    results = run_experiment(args.levels, args.configs, limit=args.limit)
    table = report(results)
    print(table.to_string(index=False))

    print(f"итого потрачено ${ledger.total:.4f}")
    print(ledger.table().to_string())