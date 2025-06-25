import kfp
import json
import mlrun
import kfp.dsl as dsl
from kfp.dsl import component

@component
def sequential_deployment(
    configs_json: str,
    dataset: str,
):
    # Get the project:
    project = mlrun.get_current_project()
    spin_up_function = project.get_function("spin-up-nims", ignore_cache=True)
    evaluate_function = project.get_function("evaluate", ignore_cache=True)
    customize_function = project.get_function("customize", ignore_cache=True)
    shutdown_function = project.get_function("shutdown-deployment", ignore_cache=True)

    prev_result = dataset
    configs = json.loads(configs_json)

    for i, config in enumerate(configs):
        spin_up_result = project.run_function(
            spin_up_function,
            name=f"spin-up-nim-{i}",
            inputs={"previous_result": prev_result},
            params={"nim_config": config},
            returns=["previous_result: file"],
        )

        base_eval_result = project.run_function(
            evaluate_function,
            name=f"base-evaluate-{i}",
            handler="run_base_eval",
            inputs={
                "previous_result": spin_up_result.outputs["previous_result"],
            },
            returns=["previous_result: file"],
        )

        icl_eval_result = project.run_function(
            evaluate_function,
            name=f"icl-evaluate-{i}",
            handler="run_icl_eval",
            inputs={
                "previous_result": spin_up_result.outputs["previous_result"],
            },
            returns=["previous_result: file"],
            local=False,
        )

        customize_result = project.run_function(
            customize_function,
            name=f"customize-{i}",
            inputs={
                "previous_result": spin_up_result.outputs["previous_result"],
            },
            returns=["previous_result: file"],
        )

        customization_eval_result = project.run_function(
            evaluate_function,
            name=f"customize-evaluate-{i}",
            handler="run_customization_eval",
            inputs={
                "previous_result": customize_result.outputs["previous_result"],
            },
            returns=["previous_result: file"],
        )

        shutdown_result = project.run_function(
            shutdown_function,
            name=f"shutdown-deployment-{i}",
            inputs={
                "base_eval_result": base_eval_result.outputs["previous_result"],
                "icl_eval_result": icl_eval_result.outputs["previous_result"],
                "customization_eval_result": customization_eval_result.outputs["previous_result"],
            },
            returns=["previous_result: file"],
        )

        prev_result = shutdown_result.outputs["previous_result"]

@dsl.pipeline()
def pipeline(
    workload_id: str,
    client_id: str,
    configs: list,
):
    # Get the project:
    project = mlrun.get_current_project()

    create_dataset_function = project.get_function("create-dataset", ignore_cache=True)
    create_dataset_result = project.run_function(
        create_dataset_function,
        params={
            "workload_id": workload_id,
            "client_id": client_id,
        },
        returns=["previous_result: file"],
    )

    wait_for_llm_as_judge_function = project.get_function(
        "wait-for-llm-as-a-judge", ignore_cache=True
    )
    wait_result = project.run_function(
        wait_for_llm_as_judge_function,
        inputs={
            "previous_result": create_dataset_result.outputs["previous_result"],
        },
        returns=["previous_result: file"],
    )

    sequential_deployment(
        configs_json=json.dumps(configs),
        dataset=wait_result.outputs["previous_result"],
    )
