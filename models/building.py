from pydantic import BaseModel
 
 
class BuildingElement(BaseModel):
    type: str        # e.g. "wall", "slab", "column"
    material: str    # e.g. "concrete", "brick"
    volume: float    # in cubic metres
    count: int = 1
 
 
class BuildingData(BaseModel):
    name: str                               # project / building name
    building_type: str                      # e.g. "residential", "commercial"
    floors: int
    total_area: float                       # m²
    elements: list[BuildingElement] = []
    notes: str = ""                         # any extra info from the plugin
 