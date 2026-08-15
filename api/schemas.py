from __future__ import annotations

from pydantic import BaseModel, Field


class PricePredictionRequest(BaseModel):
    manufacturer: str = Field(default="unknown", min_length=1)
    model: str = Field(default="other", min_length=1)
    fuel: str = Field(default="unknown", min_length=1)
    title_status: str = Field(default="unknown", min_length=1)
    transmission: str = Field(default="unknown", min_length=1)
    drive: str = Field(default="unknown", min_length=1)
    type: str = Field(default="unknown", min_length=1)
    paint_color: str = Field(default="unknown", min_length=1)
    state: str = Field(default="unknown", min_length=1)
    odometer: float = Field(..., ge=0)
    car_age: float = Field(..., ge=0)


class PricePredictionResponse(BaseModel):
    predicted_price: float
