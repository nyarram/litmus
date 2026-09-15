"""FastAPI app factory for the litmus dashboard."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from litmus.dashboard import store

_templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


def create_app(*, reports_dir: Path | str = "reports", jaeger_url: str | None = None) -> FastAPI:
    """Build the dashboard app.

    ``reports_dir``: directory of eval report JSON files.
    ``jaeger_url``: base URL of the Jaeger UI; case trace ids link there.
    """
    reports_path = Path(reports_dir)
    app = FastAPI(title="litmus dashboard")

    def _reports():
        return store.load_reports(reports_path)

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request):
        reports = _reports()
        return _templates.TemplateResponse(
            request,
            "index.html",
            {
                "request": request,
                "reports": reports,
                "scorers": store.scorer_names(reports),
            },
        )

    @app.get("/runs/{run_name}", response_class=HTMLResponse)
    def run_detail(request: Request, run_name: str):
        reports = _reports()
        report = next((r for r in reports if r.run_name == run_name), None)
        if report is None:
            raise HTTPException(status_code=404, detail=f"unknown run {run_name!r}")
        return _templates.TemplateResponse(
            request,
            "run.html",
            {
                "request": request,
                "report": report,
                "jaeger_url": (jaeger_url or "").rstrip("/"),
            },
        )

    @app.get("/trends", response_class=HTMLResponse)
    def trends(request: Request):
        reports = _reports()
        scorers = store.scorer_names(reports)
        series = {name: store.trend_points(reports, name) for name in scorers}
        return _templates.TemplateResponse(
            request,
            "trends.html",
            {"request": request, "scorers": scorers, "series": series},
        )

    @app.get("/api/runs", response_class=JSONResponse)
    def api_runs():
        return [
            {
                "run_name": r.run_name,
                "dataset": f"{r.dataset_name} v{r.dataset_version}",
                "created_at": r.created_at,
                "n_cases": len(r.results),
                "n_errors": r.n_errors,
                "summaries": {k: v.to_dict() for k, v in r.summaries.items()},
            }
            for r in _reports()
        ]

    return app
