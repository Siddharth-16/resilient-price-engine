from __future__ import annotations

from pydantic import BaseModel, Field


class PricePredictionRequest(BaseModel):
    manufacturer: str = Field(
        default="unknown",
        min_length=1,
    )

    model: str = Field(
        default="unknown",
        min_length=1,
    )

    fuel: str = Field(
        default="unknown",
        min_length=1,
    )

    title_status: str = Field(
        default="unknown",
        min_length=1,
    )

    transmission: str = Field(
        default="unknown",
        min_length=1,
    )

    drive: str = Field(
        default="unknown",
        min_length=1,
    )

    type: str = Field(
        default="unknown",
        min_length=1,
    )

    paint_color: str = Field(
        default="unknown",
        min_length=1,
    )

    state: str = Field(
        default="unknown",
        min_length=1,
    )

    odometer: float = Field(
        ...,
        ge=0,
        le=300_000,
    )

    year: int = Field(
        ...,
        ge=2000,
        le=2022,
    )


class PricePredictionResponse(BaseModel):
    predicted_price: float
