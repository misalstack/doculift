"""
Run FastAPI server
"""

import uvicorn
from src.api import app
from src.core import get_config


def main():
    config = get_config()
    
    uvicorn.run(
        "src.api:app",
        host=config.api.host,
        port=config.api.port,
        reload=config.debug,
        workers=config.api.workers if not config.debug else 1,
    )


if __name__ == "__main__":
    main()
