"""FastAPI web application for the eval suite.

Launch with: python -m evals.run_web
"""

import secrets
from pathlib import Path

from fastapi import FastAPI, Form, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from evals.report import LABEL_CLASSES, SENDER_TYPES, format_thread_label
from evals.web_auth import SECRET, AuthMiddleware, is_authenticated
from evals.web_data import (
    compare_runs,
    filter_runs,
    list_runs,
    load_run_detail,
    load_thinking_sidecar,
    unique_values,
)

app = FastAPI(title="Email Labeler Eval Suite")
app.add_middleware(AuthMiddleware)

TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# --- Jinja2 filters ---

def _format_pct(value) -> str:
    if value is None:
        return "N/A"
    return f"{value:.1%}"


def _format_duration(value) -> str:
    if value is None or value == 0:
        return "-"
    return f"{value:.2f}s"


templates.env.filters["pct"] = _format_pct
templates.env.filters["duration"] = _format_duration
templates.env.globals["SENDER_TYPES"] = SENDER_TYPES
templates.env.globals["LABEL_CLASSES"] = LABEL_CLASSES

RESULTS_DIR = Path(__file__).parent / "results"


# --- Routes ---

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, error: str = ""):
    if is_authenticated(request):
        return RedirectResponse(url="/", status_code=302)
    return templates.TemplateResponse("login.html", {"request": request, "error": error})


@app.post("/login")
async def login_submit(request: Request, password: str = Form(...)):
    if secrets.compare_digest(password, SECRET):
        response = RedirectResponse(url="/", status_code=302)
        response.set_cookie("eval_session", SECRET, httponly=True, samesite="lax", max_age=86400 * 7)
        return response
    return templates.TemplateResponse("login.html", {"request": request, "error": "Invalid password"})


@app.get("/logout")
async def logout():
    response = RedirectResponse(url="/login", status_code=302)
    response.delete_cookie("eval_session")
    return response


@app.get("/", response_class=HTMLResponse)
async def runs_list(
    request: Request,
    cloud_model: str = Query(None),
    local_model: str = Query(None),
    stages: str = Query(None),
    tag: str = Query(None),
):
    """List all eval runs with optional filtering."""
    all_runs = list_runs(RESULTS_DIR)
    options = unique_values(all_runs)
    filtered = filter_runs(all_runs, cloud_model, local_model, stages, tag)

    return templates.TemplateResponse("runs.html", {
        "request": request,
        "runs": filtered,
        "options": options,
        "filters": {
            "cloud_model": cloud_model or "",
            "local_model": local_model or "",
            "stages": stages or "",
            "tag": tag or "",
        },
    })


@app.get("/run/{run_id}", response_class=HTMLResponse)
async def run_detail(request: Request, run_id: str):
    """Detailed view of a single eval run."""
    all_runs = list_runs(RESULTS_DIR)
    matching = [(p, m) for p, m in all_runs if m.run_id.startswith(run_id)]
    if not matching:
        return HTMLResponse("Run not found", status_code=404)

    path, _ = matching[0]
    data = load_run_detail(path)
    cot_map = load_thinking_sidecar(path)
    context = data["context"]

    # Build enriched prediction rows
    rows = []
    for pred in data["predictions"]:
        cot = cot_map.get(pred.thread_id)
        label = format_thread_label(pred.thread_id, context, max_subject=50)
        correct = (
            pred.error is None
            and pred.sender_type_correct is not False
            and pred.label_correct is not False
        )
        rows.append({
            "pred": pred,
            "label": label,
            "cot": cot,
            "correct": correct,
        })

    return templates.TemplateResponse("run_detail.html", {
        "request": request,
        "meta": data["meta"],
        "metrics": data["metrics"],
        "avg_duration": data["avg_duration"],
        "rows": rows,
    })


@app.get("/compare", response_class=HTMLResponse)
async def compare_view(
    request: Request,
    baseline: str = Query(""),
    compare: list[str] = Query(None),
):
    """Compare a baseline run against one or more others."""
    all_runs = list_runs(RESULTS_DIR)

    # If no baseline selected yet, show the selection form
    if not baseline:
        return templates.TemplateResponse("compare.html", {
            "request": request,
            "runs": all_runs,
            "data": None,
        })

    # Find baseline
    baseline_matches = [p for p, m in all_runs if m.run_id.startswith(baseline)]
    if not baseline_matches:
        return HTMLResponse("Baseline run not found", status_code=404)

    compare_ids = compare or []
    compare_paths = []
    for rid in compare_ids:
        matches = [p for p, m in all_runs if m.run_id.startswith(rid)]
        if matches:
            compare_paths.append(matches[0])

    if not compare_paths:
        return templates.TemplateResponse("compare.html", {
            "request": request,
            "runs": all_runs,
            "data": None,
            "error": "Select at least one run to compare against the baseline.",
        })

    data = compare_runs(baseline_matches[0], compare_paths)

    return templates.TemplateResponse("compare.html", {
        "request": request,
        "runs": all_runs,
        "data": data,
    })
