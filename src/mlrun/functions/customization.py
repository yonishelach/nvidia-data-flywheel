import mlrun
from src.api.models import TaskResult
from src.tasks.tasks import start_customization as customization_task


def start_customization(
    context: mlrun.MLClientCtx, previous_result: dict
) -> TaskResult:
    """
    Start the customization process for a given task result.

    :param context: MLRun context.
    :param previous_result: Previous task result containing necessary configurations.
    :return: Updated TaskResult with customization details.
    """
    previous_result = TaskResult(**previous_result)
    return customization_task.run(previous_result=previous_result)
