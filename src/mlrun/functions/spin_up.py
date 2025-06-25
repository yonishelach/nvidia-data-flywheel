import mlrun
from src.api.models import TaskResult
from src.tasks.tasks import initialize_db_manager, spin_up_nim as spin_up_task


def spin_up_nim(
    context: mlrun.MLClientCtx, previous_result: dict, nim_config: dict
) -> dict:
    """
    Spin up a NIM for the given previous result.

    :param context:         MLRun context.
    :param previous_result: Previous task result containing necessary configurations.
    :param nim_config:      Configuration for the NIM to be spun up.

    :return: Updated TaskResult with NIM configuration.
    """
    initialize_db_manager()
    previous_result = TaskResult(**previous_result)
    print("--" * 50)
    print("Spinning up NIM with configuration:", nim_config)
    print("--" * 50)
    return spin_up_task.run(previous_result=previous_result, nim_config=nim_config)
