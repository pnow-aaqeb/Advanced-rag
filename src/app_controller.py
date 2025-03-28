from nest.core import Controller, Get, Post
from .app_service import AppService


@Controller("/")
class AppController:

    def __init__(self, service: AppService):
        self.service = service

    @Get("/")
    def get_app_info(self):
        return self.service.get_app_info()
    @Get("/health")
    def get_health_status(self):
        return {
            "status":"healthy",
            "running":"8000"
        }
