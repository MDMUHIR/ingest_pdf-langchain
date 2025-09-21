from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
import uvicorn
from pathlib import Path

# Import our Legal Bee RAG agent
from app.legal_bee import process_query

# Create FastAPI app
app = FastAPI(title="Legal Bee - Bangladeshi Law Assistant")

# Set up templates directory
templates_dir = Path(__file__).parent.parent / "templates"
templates_dir.mkdir(exist_ok=True)
templates = Jinja2Templates(directory=str(templates_dir))
 
# Set up static files directory
static_dir = Path(__file__).parent.parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# API models
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    query: str
    language: str
    category: str
    response: str

# API endpoints
@app.post("/api/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """Process a legal query and return the response."""
    result = process_query(request.query)
    return result

# Web UI routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the home page."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/", response_class=HTMLResponse)
async def process_form(request: Request, query: str = Form(...)):
    """Process the form submission and render the response."""
    result = process_query(query)
    return templates.TemplateResponse(
        "index.html", 
        {
            "request": request, 
            "query": result["query"],
            "response": result["response"],
            "language": result["language"],
            "category": result["category"]
        }
    )

# Run the app
if __name__ == "__main__":
    uvicorn.run("app.api:app", host="0.0.0.0", port=8000, reload=True)