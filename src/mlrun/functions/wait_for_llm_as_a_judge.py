from src.api.models import TaskResult
from src.tasks.tasks import initialize_db_manager, wait_for_llm_as_judge as judge_task

import mlrun

def wait_for_llm_as_judge(context: mlrun.MLClientCtx, previous_result: dict) -> dict:
    """
    Wait for the LLM to be ready as a judge.

    :param context: MLRun context.
    :param previous_result: Previous task result containing LLM judge configuration.
    :return: Updated TaskResult with LLM judge status.
    """
    db_manager = initialize_db_manager()
    print("List all collections in the database:")
    documents = db_manager._db["flywheel_runs"].find()
    for document in documents:
        print(document)
    print("------")
    print("previous_result:", previous_result)
    print("------")
    previous_result = TaskResult(**previous_result)
    return judge_task.run(previous_result)
