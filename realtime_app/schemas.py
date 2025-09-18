from typing import List, Optional, Tuple

from pydantic import BaseModel

class PredictRequest(BaseModel):
    image_base64: str
    detect_face: bool = True

class Probability(BaseModel):
    label: str
    probability: float

class PredictResponse(BaseModel):
    top_label: str
    probabilities: List[Probability]
    face_box: Optional[Tuple[int, int, int, int]] = None
