# from datetime import datetime
# from src.lib.integration.record_exporter import RecordExporter
# from src.lib.flywheel.cancellation import FlywheelCancelledError, check_cancellation
# from src.lib.flywheel.util import identify_workload_type
# from src.lib.integration.dataset_creator import DatasetCreator
# from src.log_utils import setup_logging
# from src.api.db_manager import TaskDBManager
# from src.api.models import FlywheelRun, NIMRunStatus
# from src.config import settings
#
# import mlrun
#
# db_manager = TaskDBManager()
# logger = setup_logging("create_dataset")
#
#
# def create_dataset(
#     context: mlrun.MLClientCtx,
#     workload_id: str,
#     client_id: str,
# ):
#     flywheel_run = FlywheelRun(
#         workload_id=workload_id,
#         client_id=client_id,
#         started_at=datetime.utcnow(),
#         num_records=0,  # Will be updated when datasets are created
#         nims=[],
#     )
#     flywheel_run_id = flywheel_run.id
#     try:
#         _check_cancellation(flywheel_run_id, raise_error=True)
#
#         # The record exporter is used to export the records from the database.
#         # The records are exported based on the split configuration.
#         # this uses the client_id and workload_id to get the records from the database.
#         records = RecordExporter().get_records(client_id, workload_id, settings.data_split_config)
#
#         # The workload type is identified based on the records.
#         # This is used to determine the type of evaluation to be run.
#         workload_type = identify_workload_type(records)
#
#         # The dataset creator is used to create the datasets.
#         # This validates to ensures that the datasets are created in the correct format for the evaluation and customization.
#         datasets = DatasetCreator(
#             records,
#             flywheel_run_id,
#             "",
#             workload_id,  # Using empty prefix for now
#             split_config=settings.data_split_config,  # Pass the split config to DatasetCreator
#         ).create_datasets(workload_type)
#
#         return {
#             "workload_id": workload_id,
#             "flywheel_run_id": str(flywheel_run_id),
#             "client_id": client_id,
#             "workload_type": workload_type,
#             "datasets": datasets,
#         }
#
#     except Exception as e:
#         error_msg = f"Error creating datasets: {e!s}"
#         logger.error(error_msg)
#         # Update flywheel run with error via the DB manager
#         db_manager.mark_flywheel_run_error(
#             flywheel_run_id, error_msg, finished_at=datetime.utcnow()
#         )
#         # Update all the NIM runs to error
#         status = (
#             NIMRunStatus.CANCELLED if isinstance(e, FlywheelCancelledError) else NIMRunStatus.FAILED
#         )
#         db_manager.mark_all_nims_status(flywheel_run_id, status, error_msg=str(e))
#         # Return a TaskResult so that downstream tasks can gracefully short-circuit
#         raise e
#
# def _check_cancellation(flywheel_run_id, raise_error=False):
#     try:
#         check_cancellation(flywheel_run_id)
#     except FlywheelCancelledError as e:
#         logger.info(f"Flywheel run cancelled: {e}")
#         if raise_error:
#             raise e
#         return True
#     return False

import mlrun
from datetime import datetime
from src.api.models import DataSplitConfig, FlywheelRun, TaskResult
from src.tasks.tasks import create_datasets
from src.lib.nemo.llm_as_judge import LLMAsJudge

def create_dataset(
    context: mlrun.MLClientCtx,
    workload_id: str,
    client_id: str,
    data_split_config: dict = None,
):
    """
    Create datasets for a given workload and client ID.

    :param context:           MLRun context.
    :param workload_id:       ID of the workload.
    :param client_id:         ID of the client.
    :param data_split_config: Configuration for data splitting.
                              If None, will use the default configuration.

    :return: A JSON representation of the TaskResult containing the created datasets and other metadata.
    """
    llm_as_judge = LLMAsJudge()
    llm_as_judge_cfg = llm_as_judge.config
    split_config = DataSplitConfig(**data_split_config) if data_split_config else None
    flywheel_run = FlywheelRun(
        workload_id=workload_id,
        client_id=client_id,
        started_at=datetime.utcnow(),
        num_records=0,  # Will be updated when datasets are created
        nims=[],
    )
    flywheel_run_id = flywheel_run.id
    previous_result = TaskResult(
        workload_id=workload_id,
        flywheel_run_id=flywheel_run_id,
        client_id=client_id,
        error=None,  # Reset any previous errors
        datasets={},
        llm_judge_config=llm_as_judge_cfg,
        data_split_config=split_config,
    )
    result = create_datasets(previous_result=previous_result)
    return result.model_dump(mode="json")
