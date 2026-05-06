import math
from typing import Generic, List, Optional, TypeVar

from pydantic import BaseModel, model_validator

T = TypeVar("T")


class ApiResponse(BaseModel, Generic[T]):
    success: bool = True
    data: Optional[T] = None
    message: Optional[str] = None

    @classmethod
    def ok(cls, data: T, message: Optional[str] = None) -> "ApiResponse[T]":
        return cls(success=True, data=data, message=message)

    @classmethod
    def error(cls, message: str) -> "ApiResponse[None]":
        return cls(success=False, data=None, message=message)


class PageResponse(BaseModel, Generic[T]):
    items: List[T]
    total: int
    page: int
    size: int
    pages: int = 0

    @model_validator(mode="after")
    def compute_pages(self) -> "PageResponse[T]":
        if self.size > 0:
            self.pages = math.ceil(self.total / self.size)
        else:
            self.pages = 0
        return self
