from pydantic import BaseModel
 
 
class ScheduleTask(BaseModel):
    task_name: str
    duration_days: int
    depends_on: list[str] = []     # names of tasks that must finish first
    workers_required: int
    description: str = ""
 
 
class ScheduleResponse(BaseModel):
    project_name: str
    total_duration_days: int
    tasks: list[ScheduleTask]
    notes: str = ""
 