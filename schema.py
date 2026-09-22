from pydantic import BaseModel, ConfigDict
from typing import Optional


class Education(BaseModel):
    model_config = ConfigDict(extra="ignore")
    institution: str
    degree: str
    year: Optional[str] = None


class Employment(BaseModel):
    model_config = ConfigDict(extra="ignore")
    company: str
    title: str
    start: Optional[str] = None
    end: Optional[str] = None
    summary: Optional[str] = None


class Resume(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: str
    emails: list[str] = []
    phones: list[str] = []
    total_years_experience: Optional[float] = None
    education: list[Education] = []
    employment: list[Employment] = []
    skills: list[str] = []
    certifications: list[str] = []
    location: Optional[str] = None


class ResumeWithConfidence(BaseModel):
    model_config = ConfigDict(extra="ignore")
    data: Resume
    confidence: dict[str, float] = {}