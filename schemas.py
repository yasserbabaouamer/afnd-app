

from pydantic import BaseModel


class TextSchema(BaseModel):
    text: str


class TextAnalysis(BaseModel):
    prediction: str
    confidence: float
    word_count: int
    sent_count: int
    polarity: float
    subjectivity: float
    count_per: int
    count_loc: int
    count_org: int
    count_noun: int
    count_verb: int
    count_adj: int
    count_adv: int
    most_frequent_words: list[dict[str, int]]
