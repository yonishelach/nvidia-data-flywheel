import mlrun
from src.api.models import TaskResult
from src.tasks.tasks import (
    initialize_db_manager,
    run_base_eval as base_eval_task,
    run_customization_eval as customization_eval_task,
    run_icl_eval as icl_eval_task
)


def run_base_eval(context: mlrun.MLClientCtx, previous_result: dict) -> dict:
    """
    Run the base evaluation for a given task result.

    :param context:         MLRun context.
    :param previous_result: Previous task result containing necessary configurations.

    :return: Updated TaskResult with base evaluation results.
    """
    initialize_db_manager()
    previous_result = TaskResult(**previous_result)
    return base_eval_task.run(previous_result=previous_result)


def run_icl_eval(context: mlrun.MLClientCtx, previous_result: dict) -> dict:
    """
    Run the in-context learning evaluation for a given task result.

    :param context:         MLRun context.
    :param previous_result: Previous task result containing necessary configurations.

    :return: Updated TaskResult with in-context learning evaluation results.
    """
    initialize_db_manager()
    previous_result = TaskResult(**previous_result)
    return icl_eval_task.run(previous_result=previous_result)


def run_customization_eval(context: mlrun.MLClientCtx, previous_result: dict) -> dict:
    """
    Run the customization evaluation for a given task result.

    :param context:         MLRun context.
    :param previous_result: Previous task result containing necessary configurations.

    :return: Updated TaskResult with customization evaluation results.
    """
    initialize_db_manager()
    previous_result = TaskResult(**previous_result)
    return customization_eval_task.run(previous_result=previous_result)
