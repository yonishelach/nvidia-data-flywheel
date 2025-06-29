import plotly.graph_objects as go
from plotly.subplots import make_subplots

import mlrun
from src.tasks.tasks import initialize_db_manager
from src.tasks.tasks import shutdown_deployment as shutdown_task


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

    _create_plot_artifact(
        context,
        base_eval_result,
        icl_eval_result,
        customization_eval_result,
    )

    return shutdown_task.run(previous_results=previous_results)


def _create_plot_artifact(
    context: mlrun.MLClientCtx,
    base_eval_result: dict,
    icl_eval_result: dict,
    customization_eval_result: dict,
):
    # Extract scores from each dictionary
    base_eval_name, base_scores = _extract_evaluation_scores(base_eval_result)
    custom_eval_name, custom_scores = _extract_evaluation_scores(icl_eval_result)
    icl_eval_name, icl_scores = _extract_evaluation_scores(customization_eval_result)

    # Get metric names (assuming all evaluations have the same metrics)
    metrics = list(base_scores.keys())

    # Create the grouped bar chart
    fig = go.Figure()

    # Add traces for each evaluation
    fig.add_trace(
        go.Bar(
            name="BASE-EVAL",
            x=metrics,
            y=list(base_scores.values()),
            marker_color="#1f77b4",
        )
    )

    fig.add_trace(
        go.Bar(
            name="CUSTOMIZED-EVAL",
            x=metrics,
            y=list(custom_scores.values()),
            marker_color="#ff7f0e",
        )
    )

    fig.add_trace(
        go.Bar(
            name="ICL-EVAL",
            x=metrics,
            y=list(icl_scores.values()),
            marker_color="#2ca02c",
        )
    )

    # Update layout
    fig.update_layout(
        title=f'Evaluation Results for {base_eval_result["nim"]["model_name"]}',
        xaxis_title="Metrics",
        yaxis_title="Score",
        barmode="group",
        yaxis=dict(range=[0, 1]),
        xaxis=dict(tickangle=-45),
        legend=dict(x=0.7, y=1),
        height=600,
        width=900,
    )
    context.log_artifact(
        "evaluation-results-plot",
        body=fig.to_html(),  # convert object for logging an html file
        format="html",
    )


def _extract_evaluation_scores(data_dict):
    """Extract scores from evaluation dictionary"""
    evaluations = data_dict.get("evaluations", {})

    # Get the first (and only) evaluation key
    eval_key = list(evaluations.keys())[0]
    scores = evaluations[eval_key].get("scores", {})

    return eval_key, scores
