from datetime import datetime

import mlrun
from src.api.models import FlywheelRun, TaskResult
from src.tasks.tasks import create_datasets, initialize_db_manager, initialize_workflow


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
    db_manager = initialize_db_manager()
    flywheel_run = FlywheelRun(
        workload_id=workload_id,
        client_id=client_id,
        started_at=datetime.utcnow(),
        num_records=0,  # Will be updated when datasets are created
        nims=[],
    )
    result = db_manager._db.flywheel_runs.insert_one(flywheel_run.to_mongo())
    flywheel_run.id = str(result.inserted_id)
    result = initialize_workflow.run(
        workload_id=workload_id,
        flywheel_run_id=flywheel_run.id,
        client_id=client_id,
        data_split_config=data_split_config if data_split_config else None,
    )
    previous_result = TaskResult(**result)
    return create_datasets(previous_result=previous_result)
