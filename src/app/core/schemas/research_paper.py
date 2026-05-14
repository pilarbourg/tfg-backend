from dataclasses import dataclass

@dataclass
class ResearchPaper:
    pmid: str
    pmcid: str | None
    doi: str | None
    title: str
    year: str
    abstract: str | None
    has_full_text: bool