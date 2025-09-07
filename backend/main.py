import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .router import router
from dotenv import load_dotenv

load_dotenv()

# 👇 This must exist
app = FastAPI(title="RAG Web Chat Backend")

# Allow Streamlit (localhost:8501) to access API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include all routes
app.include_router(router)

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
