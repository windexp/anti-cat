"""API 관련 모듈"""
from .frigate_api import FrigateAPI
from .dashboard_api import router as dashboard_router

__all__ = ["FrigateAPI", "dashboard_router"]
