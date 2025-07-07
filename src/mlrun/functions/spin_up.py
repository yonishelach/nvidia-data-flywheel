import json
import mlrun
import requests

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
    if not nim_application.is_deployed():
        nim_application.deploy()
    else:
        context.logger.info(f"NIM {model_name} is already deployed.")
    initialize_db_manager()
    # Add NIM configuration to the previous result
    previous_result["nim"] = nim_config
    # add deployment config to nemo:
    payload = {
        "name": formatted_model_name,
        "external_endpoint": {
            "host_url": f"http://{nim_application.get_url()}:32221",
            "enabled_models": [
                "meta/llama-3.2-1b-instruct"
            ]
        },
    }
    resp = requests.post(f"http://{settings.nmp_config.nemo_base_url}/v1/deployment/configs", json=payload)
    if resp.status_code != 200:
        context.logger.error(f"Failed to add deployment config: {resp.text}")
        raise Exception(f"Failed to add deployment config: {resp.text}")
    context.logger.info(f"Deployment config added successfully: {resp.json()}")

    return previous_result
