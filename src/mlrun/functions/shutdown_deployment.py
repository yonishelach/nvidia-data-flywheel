from src.tasks.tasks import shutdown_deployment as shutdown_task
from src.api.models import TaskResult

import mlrun

def shutdown_deployment(context: mlrun.MLClientCtx, previous_result: dict) -> dict:
    """
    Shutdown the deployment for a given task result.

    :param context:         MLRun context.
    :param previous_result: Previous task result containing necessary configurations.

    :return: Updated TaskResult with shutdown status.
    """
    previous_result = TaskResult(**previous_result)
    return shutdown_task(previous_result=previous_result)
