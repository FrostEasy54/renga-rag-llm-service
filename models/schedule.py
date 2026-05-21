from pydantic import BaseModel
 
 
class ScheduleTask(BaseModel):
    task_name: str
    duration_days: int
    depends_on: list[str] = []
    workers_required: int
    work_shifts: int = 2
    description: str = ""
 
 
class ScheduleResponse(BaseModel):
    project_name: str
    total_duration_days: int
    tasks: list[ScheduleTask]
    notes: str = ""
    rag_context_used: bool = False