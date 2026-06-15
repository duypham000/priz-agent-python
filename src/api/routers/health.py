from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
@router.get("/api/v2/agent/health")
async def health():
    return {"status": "ok", "version": "0.1.0"}
