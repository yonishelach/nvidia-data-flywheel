import json
import mlrun
import requests
from time import sleep
from src.tasks.tasks import initialize_db_manager
from src.mlrun.functions.nim_application import NIMApplication
from src.config import settings

def spin_up_nim(
    context: mlrun.MLClientCtx, previous_result: dict, nim_config: dict, project_name: str = None
) -> dict:
    """
    Spin up a NIM for the given previous result.

    :param context:         MLRun context.
    :param previous_result: Previous task result containing necessary configurations.
    :param nim_config:      Configuration for the NIM to be spun up.
    :param project_name:    Name of the project (optional).

    :return: Updated TaskResult with NIM configuration.
    """
    if isinstance(nim_config, str):
        # If nim_config is a string, assume it's a JSON string and parse it
        nim_config = json.loads(nim_config)
    model_name = nim_config.get("model_name")
    formatted_model_name = model_name.replace('/', '-')

    nim_application = NIMApplication(
        name=f"nim-{formatted_model_name}",
        model_name=model_name,
        image_name=nim_config.get("image_name"),
        tag=nim_config.get("tag", "latest"),
        project_name=project_name,
    )
    nim_application.deploy(force_redeploy=True)
    initialize_db_manager()
    # Add NIM configuration to the previous result
    previous_result["nim"] = nim_config
    # Create evaluation targets:
    payload = {
        "type": "model",
        "name": formatted_model_name,
        "namespace": settings.nmp_config.nmp_namespace,
        "model": {
            "api_endpoint": {
                "url": f"http://{nim_application.get_url()}/v1/chat/completions",
                "model_id": model_name,
                "format": "openai"
            }
        }
    }
    # Add the NIM model to the evaluation targets:
    previous_result["evaluation_targets"] = [f"{settings.nmp_config.nmp_namespace}/{formatted_model_name}"]
    previous_result["mlrun_function"] = f"nim-{formatted_model_name}"
    response = requests.post(f"{settings.nmp_config.nemo_base_url}/v1/evaluation/targets", json=payload)
    if not response.ok:
        context.logger.error(f"Failed to add deployment config: {response.text}")
        raise Exception(f"Failed to add deployment config: {response.text}")
    context.logger.info(f"Deployment config added successfully: {response.json()}")
    # check that the nim_application is deployed by checking api endpoint:
    timeout = 300
    while timeout > 0:
        try:
            response = requests.get(f"http://{nim_application.get_url()}/v1/health/ready")
            if response.ok:
                context.logger.info(f"NIM application {nim_application._name} is up and running.")
                break
        except requests.exceptions.RequestException as e:
            context.logger.warning(f"Waiting for NIM application {nim_application._name} to be ready: {e}")
        timeout -= 5
        sleep(5)
    return previous_result
