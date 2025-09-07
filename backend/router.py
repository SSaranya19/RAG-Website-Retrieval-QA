from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from pydantic import BaseModel
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
import openai

from .engine import engine
from .models import ChatRequest, ChatResponse, Answer, IndexRequest

load_dotenv()
router = APIRouter()

# ---------------- JWT CONFIG ----------------
SECRET_KEY = os.getenv("SECRET_KEY", "supersecret")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

VALID_USERNAME = os.getenv("VALID_USERNAME", "admin")
VALID_PASSWORD = os.getenv("VALID_PASSWORD", "admin123")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/login")

class Token(BaseModel):
    access_token: str
    token_type: str

class User(BaseModel):
    username: str

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        return User(username=username)
    except JWTError:
        raise credentials_exception

# ---------------- LOGIN ----------------
@router.post("/api/v1/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    if form_data.username != VALID_USERNAME or form_data.password != VALID_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    token = create_access_token(data={"sub": form_data.username}, expires_delta=access_token_expires)
    return {"access_token": token, "token_type": "bearer"}

# ---------------- INDEX ----------------
@router.post("/api/v1/index")
def index_urls(req: IndexRequest, current_user: User = Depends(get_current_user)):
    """
    Index new URLs. If mode=replace → reset index before adding.
    """
    try:
        results = []
        if req.mode == "replace":
            engine.reset_index()

        for url in req.urls:
            try:
                r = engine.index_url(url)
                results.append({"url": url, "status": "success"})
            except Exception as e:
                results.append({"url": url, "status": "error", "detail": str(e)})

        return {"indexed": results, "mode": req.mode}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/v1/index")
def list_indexed(current_user: User = Depends(get_current_user)):
    """
    Return list of currently indexed URLs.
    """
    try:
        return {"indexed": engine.list_indexed()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------------- CHAT ----------------
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_KEY:
    openai.api_key = OPENAI_KEY

@router.post("/api/v1/chat", response_model=ChatResponse)
def chat(req: ChatRequest, top_k: int = 5, current_user: User = Depends(get_current_user)):
    last_user = next((m.content for m in reversed(req.messages) if m.role == "user"), None)
    if not last_user:
        raise HTTPException(status_code=400, detail="No user message found")

    hits = engine.retrieve(last_user, top_k=top_k)

    grouped = {}
    for h in hits:
        url = h["metadata"].get("url")
        if not url:
            continue
        grouped.setdefault(url, []).append(h["document"])

    answers_by_url = {}
    if OPENAI_KEY:
        for url, docs in grouped.items():
            context = "\n\n".join(docs)
            system = (
                "You are a helpful assistant. Use ONLY the following excerpts to answer. "
                "Keep it concise. Do not add extra sources."
            )
            user_prompt = f"Question: {last_user}\n\nExcerpts from {url}:\n{context}"
            try:
                resp = openai.ChatCompletion.create(
                    model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=400,
                    temperature=0.0,
                )
                answers_by_url[url] = resp["choices"][0]["message"]["content"].strip()
            except Exception as e:
                answers_by_url[url] = f"(Error: {e})"
    else:
        for url, docs in grouped.items():
            answers_by_url[url] = " ".join([d[:400] for d in docs])

    final_answer = ""
    for url, ans in answers_by_url.items():
        final_answer += f"**From {url}:**\n{ans}\n\n🔗 [Read more here]({url})\n\n"

    return ChatResponse(answer=Answer(content=final_answer.strip()), citations=[])
