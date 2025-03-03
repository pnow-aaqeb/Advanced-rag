import uvicorn
from src.app_module import http_server as app

if __name__ == '__main__':
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True
    )
    
