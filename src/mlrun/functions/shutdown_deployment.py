import mlrun
from src.api.models import TaskResult
from src.tasks.tasks import initialize_db_manager, shutdown_deployment as shutdown_task


def shutdown_deployment(context: mlrun.MLClientCtx, previous_result: dict) -> dict:
    """
    Shutdown the deployment for a given task result.

    :param context:         MLRun context.
    :param previous_result: Previous task result containing necessary configurations.

    :return: Updated TaskResult with shutdown status.
    """
    initialize_db_manager()
    previous_result = TaskResult(**previous_result)
    return shutdown_task.run(previous_result=previous_result)
