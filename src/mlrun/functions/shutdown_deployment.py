import mlrun
from src.tasks.tasks import initialize_db_manager
from src.tasks.tasks import shutdown_deployment as shutdown_task, _extract_previous_result


def shutdown_deployment(
    context: mlrun.MLClientCtx,
    base_eval_result: dict,
    icl_eval_result: dict,
    customization_eval_result: dict,
    project_name: str = None
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
    if project_name:
        function_name = base_eval_result["mlrun_function"]
        project = mlrun.get_or_create_project(project_name)
        project.delete_function(function_name, delete_from_db=True)
        previous_result = _extract_previous_result(
            previous_results,
            validator=lambda r: getattr(r, "nim", None) is not None,
            error_msg="No valid TaskResult with NIM config found in results",
        )
        return previous_result.model_dump()
    else:
        return shutdown_task.run(previous_results=previous_results)
