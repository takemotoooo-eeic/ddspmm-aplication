from pydantic import BaseModel, StrictFloat

class Note(BaseModel):
    start: StrictFloat
    frequency: StrictFloat
    duration: StrictFloat


class AlignedMidi(BaseModel):
    notes: list[Note]
