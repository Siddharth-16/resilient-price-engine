from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class PricePredictionRequest(BaseModel):
    manufacturer: Optional[str] = Field(default="unknown")
    model: Optional[str] = Field(default="other")
    fuel: Optional[str] = Field(default="unknown")
    title_status: Optional[str] = Field(default="unknown")
    transmission: Optional[str] = Field(default="unknown")
    drive: Optional[str] = Field(default="unknown")
    type: Optional[str] = Field(default="unknown")
    paint_color: Optional[str] = Field(default="unknown")
    state: Optional[str] = Field(default="unknown")
    odometer: float = Field(..., ge=0)
    car_age: float = Field(..., ge=0)


class PricePredictionResponse(BaseModel):
    predicted_price: float