from pydantic import BaseModel
 
 
class BuildingElement(BaseModel):
    type: str
    material: str
    volume: float
    count: int = 1
 
 
class BuildingData(BaseModel):
    name: str
    building_type: str
    floors: int
    total_area: float
    elements: list[BuildingElement] = []
    notes: str = ""
 