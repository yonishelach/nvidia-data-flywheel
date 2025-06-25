import mlrun
from src.tasks.tasks import initialize_db_manager, shutdown_deployment as shutdown_task


def shutdown_deployment(
    context: mlrun.MLClientCtx,
    base_eval_result: dict,
    icl_eval_result: dict,
    customization_eval_result: dict,
) -> dict:
    """
    Shutdown the deployment for a given task result.

    :param context:                   MLRun context.
    :param base_eval_result:          Base evaluation results to be included in the shutdown.
    :param icl_eval_result:           In-context learning evaluation results to be included in the shutdown.
    :param customization_eval_result: Customization evaluation results to be included in the shutdown.

    :return: Updated TaskResult with shutdown status.
    """
    initialize_db_manager()
    previous_results = [
        base_eval_result,
        icl_eval_result,
        customization_eval_result,
    ]
    return shutdown_task.run(previous_results=previous_results)
