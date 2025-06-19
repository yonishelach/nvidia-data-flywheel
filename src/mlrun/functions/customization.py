from src.tasks.tasks import initialize_db_manager, start_customization as customization_task
from src.api.models import TaskResult

import mlrun

def start_customization(context: mlrun.MLClientCtx, previous_result: dict) -> TaskResult:
    """
    Start the customization process for a given task result.

    :param context: MLRun context.
    :param previous_result: Previous task result containing necessary configurations.
    :return: Updated TaskResult with customization details.
    """
    initialize_db_manager()
    previous_result = TaskResult(**previous_result)
    return customization_task(previous_result=previous_result)