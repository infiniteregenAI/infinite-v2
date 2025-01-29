import traceback
import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from utils.constants import AVAILABLE_TOOLS

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/available-tools")
async def get_available_tools():
    """
        This endpoint returns the available tools.
        
        Returns:
            JSONResponse: The response body.
    """
    try:
        return JSONResponse(
            status_code=200,
            content={"available_tools": AVAILABLE_TOOLS}
        )
    except Exception as e:
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"message": "Internal server error"}
        )