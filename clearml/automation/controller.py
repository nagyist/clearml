import atexit
import functools
import inspect
import json
import os
import re

import six
import warnings
from copy import copy, deepcopy
from datetime import datetime
from logging import getLogger
from multiprocessing import Process, Queue
from multiprocessing.pool import ThreadPool
from threading import Thread, Event, RLock, current_thread
from time import time, sleep
from typing import Sequence, Optional, Mapping, Callable, List, Dict, Union, Tuple, Any

from attr import attrib, attrs
from pathlib2 import Path

from .job import LocalClearmlJob, RunningJob, BaseJob
from .. import Logger
from ..automation import ClearmlJob
from ..backend_api import Session
from ..backend_interface.task.populate import CreateFromFunction
from ..backend_interface.util import get_or_create_project, mutually_exclusive
from ..config import get_remote_task_id
from ..debugging.log import LoggerRoot
from ..errors import UsageError
from ..model import BaseModel, OutputModel
from ..storage.util import hash_dict
from ..task import Task
from ..utilities.process.mp import leave_process
from ..utilities.proxy_object import (
    LazyEvalWrapper,
    flatten_dictionary,
    walk_nested_dict_tuple_list,
)
from ..utilities.version import Version


class PipelineController(object):
    """
    Pipeline controller.
    Pipeline is a DAG of base tasks, each task will be cloned (arguments changed as required), executed, and monitored.
    The pipeline process (task) itself can be executed manually or by the clearml-agent services queue.
    Notice: The pipeline controller lives as long as the pipeline itself is being executed.
    """

    _tag = "pipeline"
    _project_system_tags = ["pipeline", "hidden"]
    _node_tag_prefix = "pipe:"
    _step_pattern = r"\${[^}]*}"
    _config_section = "Pipeline"
    _state_artifact_name = "pipeline_state"
    _args_section = "Args"
    _pipeline_section = "pipeline"
    _pipeline_step_ref = "pipeline"
    _runtime_property_hash = "_pipeline_hash"
    _relaunch_status_message = "Relaunching pipeline step..."
    _reserved_pipeline_names = (_pipeline_step_ref,)
    _task_project_lookup = {}
    _clearml_job_class = ClearmlJob
    _update_execution_plot_interval = 5.0 * 60
    _update_progress_interval = 10.0
    _monitor_node_interval = 5.0 * 60
    _pipeline_as_sub_project_cached = None
    _report_plot_execution_flow = dict(title="Pipeline", series="Execution Flow")
    _report_plot_execution_details = dict(title="Pipeline Details", series="Execution Details")
    _evaluated_return_values = {}  # TID: pipeline_name
    _add_to_evaluated_return_values = {}  # TID: bool
    _retries = {}  # Node.name: int
    _retries_callbacks = {}  # Node.name: Callable[[PipelineController, PipelineController.Node, int], bool]  # noqa
    _status_change_callbacks = {}  # Node.name: Callable[PipelineController, PipelineController.Node, str]
    _final_failure = {}  # Node.name: bool
    _task_template_header = CreateFromFunction.default_task_template_header
    _default_pipeline_version = "1.0.0"
    _project_section = ".pipelines"

    valid_job_status = [
        "failed",
        "cached",
        "completed",
        "aborted",
        "queued",
        "running",
        "skipped",
        "pending",
    ]

    @attrs
    class Node(object):
        # pipeline step name
        name = attrib(type=str)
        # base Task ID to be cloned and launched
        base_task_id = attrib(type=str, default=None)
        # alternative to base_task_id, function creating a Task
        task_factory_func = attrib(type=Callable, default=None)
        # execution queue name to use
        queue = attrib(type=str, default=None)
        # list of parent DAG steps
        parents = attrib(type=list, default=None)
        # execution timeout limit
        timeout = attrib(type=float, default=None)
        # Task hyper-parameters to change
        parameters = attrib(type=dict, default=None)
        # Task configuration objects to change
        configurations = attrib(type=dict, default=None)
        # Task overrides to change
        task_overrides = attrib(type=dict, default=None)
        # The actual executed Task ID (None if not executed yet)
        executed = attrib(type=str, default=None)
        # The Node Task status (cached, aborted, etc.)
        status = attrib(type=str, default="pending")
        # If True cline the base_task_id, then execute the cloned Task
        clone_task = attrib(type=bool, default=True)
        # ClearMLJob object
        job = attrib(type=ClearmlJob, default=None)
        # task type (string)
        job_type = attrib(type=str, default=None)
        # job startup timestamp (epoch ts in seconds)
        job_started = attrib(type=float, default=None)
        # job startup timestamp (epoch ts in seconds)
        job_ended = attrib(type=float, default=None)
        # pipeline code configuration section name
        job_code_section = attrib(type=str, default=None)
        # if True, this step should be skipped
        skip_job = attrib(type=bool, default=False)
        # if True this pipeline step should be cached
        cache_executed_step = attrib(type=bool, default=False)
        # List of artifact names returned by the step
        return_artifacts = attrib(type=list, default=None)
        # List of metric title/series to monitor
        monitor_metrics = attrib(type=list, default=None)
        # List of artifact names to monitor
        monitor_artifacts = attrib(type=list, default=None)
        # List of models to monitor
        monitor_models = attrib(type=list, default=None)
        # The Docker image the node uses, specified at creation
        explicit_docker_image = attrib(type=str, default=None)
        # if True, recursively parse parameters in lists, dicts, or tuples
        recursively_parse_parameters = attrib(type=bool, default=False)
        # The default location for output models and other artifacts
        output_uri = attrib(type=Union[bool, str], default=None)
        # Specify whether to create the Task as a draft
        draft = attrib(type=bool, default=False)
        # continue_behaviour dict, for private use. used to initialize fields related to continuation behaviour
        continue_behaviour = attrib(type=dict, default=None)
        # if True, the pipeline continues even if the step failed
        continue_on_fail = attrib(type=bool, default=False)
        # if True, the pipeline continues even if the step was aborted
        continue_on_abort = attrib(type=bool, default=False)
        # if True, the children of aborted steps are skipped
        skip_children_on_abort = attrib(type=bool, default=True)
        # if True, the children of failed steps are skipped
        skip_children_on_fail = attrib(type=bool, default=True)
        # the stage of the step
        stage = attrib(type=str, default=None)

        def __attrs_post_init__(self) -> None:
            if self.parents is None:
                self.parents = []
            if self.parameters is None:
                self.parameters = {}
            if self.configurations is None:
                self.configurations = {}
            if self.task_overrides is None:
                self.task_overrides = {}
            if self.return_artifacts is None:
                self.return_artifacts = []
            if self.monitor_metrics is None:
                self.monitor_metrics = []
            if self.monitor_artifacts is None:
                self.monitor_artifacts = []
            if self.monitor_models is None:
                self.monitor_models = []
            if self.continue_behaviour is not None:
                self.continue_on_fail = self.continue_behaviour.get("continue_on_fail", True)
                self.continue_on_abort = self.continue_behaviour.get("continue_on_abort", True)
                self.skip_children_on_fail = self.continue_behaviour.get("skip_children_on_fail", True)
                self.skip_children_on_abort = self.continue_behaviour.get("skip_children_on_abort", True)
                self.continue_behaviour = None

        def copy(self) -> "PipelineController.Node":
            """
            return a copy of the current Node, excluding the `job`, `executed`, fields
            :return: new Node copy
            """
            new_copy = PipelineController.Node(
                name=self.name,
                **dict(
                    (k, deepcopy(v))
                    for k, v in self.__dict__.items()
                    if k not in ("name", "job", "executed", "task_factory_func")
                ),
            )
            new_copy.task_factory_func = self.task_factory_func
            return new_copy

        def set_job_ended(self) -> None:
            if self.job_ended:
                return
            # noinspection PyBroadException
            try:
                self.job.task.reload()
                self.job_ended = self.job_started + self.job.task.data.active_duration
            except Exception:
                pass

        def set_job_started(self) -> None:
            if self.job_started:
                return
            # noinspection PyBroadException
            try:
                self.job_started = self.job.task.data.started.timestamp()
            except Exception:
                pass

    def __init__(
        self,
        name: str,
        project: str,
        version: Optional[str] = None,
        pool_frequency: float = 0.2,
        add_pipeline_tags: bool = False,
        target_project: Optional[Union[str, bool]] = True,
        auto_version_bump: Optional[bool] = None,
        abort_on_failure: bool = False,
        add_run_number: bool = True,
        retry_on_failure: Optional[
            Union[
                int,
                Callable[["PipelineController", "PipelineController.Node", int], bool],
            ]
        ] = None,  # noqa
        docker: Optional[str] = None,
        docker_args: Optional[str] = None,
        docker_bash_setup_script: Optional[str] = None,
        packages: Optional[Union[bool, str, Sequence[str]]] = None,
        repo: Optional[str] = None,
        repo_branch: Optional[str] = None,
        repo_commit: Optional[str] = None,
        always_create_from_code: bool = True,
        artifact_serialization_function: Optional[Callable[[Any], Union[bytes, bytearray]]] = None,
        artifact_deserialization_function: Optional[Callable[[bytes], Any]] = None,
        output_uri: Optional[Union[str, bool]] = None,
        skip_global_imports: bool = False,
        working_dir: Optional[str] = None,
        enable_local_imports: bool = True,
    ) -> None:
        """
        Create a new pipeline controller. The newly created object will launch and monitor the new experiments.

        :param name: Provide pipeline name (if main Task exists it overrides its name)
        :param project: Provide project storing the pipeline (if main Task exists  it overrides its project)
        :param version: Pipeline version. This version allows to uniquely identify the pipeline
            template execution. Examples for semantic versions: version='1.0.1' , version='23', version='1.2'.
            If not set, find the latest version of the pipeline and increment it. If no such version is found,
            default to '1.0.0'
        :param float pool_frequency: The pooling frequency (in minutes) for monitoring experiments / states.
        :param bool add_pipeline_tags: (default: False) if True, add `pipe: <pipeline_task_id>` tag to all
            steps (Tasks) created by this pipeline.
        :param str target_project: If provided, all pipeline steps are cloned into the target project.
            If True, pipeline steps are stored into the pipeline project
        :param bool auto_version_bump: (Deprecated) If True, if the same pipeline version already exists
            (with any difference from the current one), the current pipeline version will be bumped to a new version
            version bump examples: 1.0.0 -> 1.0.1 , 1.2 -> 1.3, 10 -> 11 etc.
        :param bool abort_on_failure: If False (default), failed pipeline steps will not cause the pipeline
            to stop immediately, instead any step that is not connected (or indirectly connected) to the failed step,
            will still be executed. Nonetheless the pipeline itself will be marked failed, unless the failed step
            was specifically defined with "continue_on_fail=True".
            If True, any failed step will cause the pipeline to immediately abort, stop all running steps,
            and mark the pipeline as failed.
        :param add_run_number: If True (default), add the run number of the pipeline to the pipeline name.
            Example, the second time we launch the pipeline "best pipeline", we rename it to "best pipeline #2"
        :param retry_on_failure: Integer (number of retries) or Callback function that returns True to allow a retry

          - Integer: In case of node failure, retry the node the number of times indicated by this parameter.
          - Callable: A function called on node failure. Takes as parameters:
            the PipelineController instance, the PipelineController.Node that failed and an int
            representing the number of previous retries for the node that failed.
            The function must return ``True`` if the node should be retried and ``False`` otherwise.
            If True, the node will be re-queued and the number of retries left will be decremented by 1.
            By default, if this callback is not specified, the function will be retried the number of
            times indicated by `retry_on_failure`.

              .. code-block:: py

                  def example_retry_on_failure_callback(pipeline, node, retries):
                      print(node.name, ' failed')
                      # allow up to 5 retries (total of 6 runs)
                      return retries < 5
        :param docker: Select the docker image to be executed in by the remote session
        :param docker_args: Add docker arguments, pass a single string
        :param docker_bash_setup_script: Add bash script to be executed
            inside the docker before setting up the Task's environment
        :param packages: Manually specify a list of required packages or a local requirements.txt file.
            Example: ["tqdm>=2.1", "scikit-learn"] or "./requirements.txt"
            If not provided, packages are automatically added.
            Use `False` to install requirements from "requirements.txt" inside your git repository
        :param repo: Optional, specify a repository to attach to the pipeline controller, when remotely executing.
            Allow users to execute the controller inside the specified repository, enabling them to load modules/script
            from the repository. Notice the execution work directory will be the repository root folder.
            Supports both git repo url link, and local repository path (automatically converted into the remote
            git/commit as is currently checkout).
            Example remote url: 'https://github.com/user/repo.git'
            Example local repo copy: './repo' -> will automatically store the remote
            repo url and commit ID based on the locally cloned copy
            Use empty string ("") to disable any repository auto-detection
        :param repo_branch: Optional, specify the remote repository branch (Ignored, if local repo path is used)
        :param repo_commit: Optional, specify the repository commit ID (Ignored, if local repo path is used)
        :param always_create_from_code: If True (default) the pipeline is always constructed from code,
            if False, pipeline is generated from pipeline configuration section on the pipeline Task itsef.
            this allows to edit (also add/remove) pipeline steps without changing the original codebase
        :param artifact_serialization_function: A serialization function that takes one
            parameter of any type which is the object to be serialized. The function should return
            a `bytes` or `bytearray` object, which represents the serialized object. All parameter/return
            artifacts uploaded by the pipeline will be serialized using this function.
            All relevant imports must be done in this function. For example:

            .. code-block:: py

                def serialize(obj):
                    import dill
                    return dill.dumps(obj)
        :param artifact_deserialization_function: A deserialization function that takes one parameter of type `bytes`,
            which represents the serialized object. This function should return the deserialized object.
            All parameter/return artifacts fetched by the pipeline will be deserialized using this function.
            All relevant imports must be done in this function. For example:

            .. code-block:: py

                def deserialize(bytes_):
                    import dill
                    return dill.loads(bytes_)
        :param output_uri: The storage / output url for this pipeline. This is the default location for output
            models and other artifacts. Check Task.init reference docs for more info (output_uri is a parameter).
            The `output_uri` of this pipeline's steps will default to this value.
        :param skip_global_imports: If True, global imports will not be included in the steps' execution when creating
            the steps from a functions, otherwise all global imports will be automatically imported in a safe manner at
             the beginning of each step’s execution. Default is False
        :param working_dir: Working directory to launch the pipeline from.
        :param enable_local_imports: If True, allow pipeline steps to import from local files
            by appending to the PYTHONPATH of each step the directory the pipeline controller
            script resides in (sys.path[0]).
            If False, the directory won't be appended to PYTHONPATH. Default is True.
            Ignored while running remotely.
        """
        if auto_version_bump is not None:
            warnings.warn(
                "PipelineController.auto_version_bump is deprecated. It will be ignored",
                DeprecationWarning,
            )
        self._nodes = {}
        self._running_nodes = []
        self._start_time = None
        self._pipeline_time_limit = None
        self._default_execution_queue = None
        self._always_create_from_code = bool(always_create_from_code)
        self._version = str(version).strip() if version else None
        if self._version and not Version.is_valid_version_string(self._version):
            raise ValueError("Setting non-semantic pipeline version '{}'".format(self._version))
        self._pool_frequency = pool_frequency * 60.0
        self._thread = None
        self._pipeline_args = dict()
        self._pipeline_args_desc = dict()
        self._pipeline_args_type = dict()
        self._args_map = dict()
        self._stop_event = None
        self._experiment_created_cb = None
        self._experiment_completed_cb = None
        self._pre_step_callbacks = {}
        self._post_step_callbacks = {}
        self._target_project = target_project
        self._add_pipeline_tags = add_pipeline_tags
        self._task = Task.current_task()
        self._step_ref_pattern = re.compile(self._step_pattern)
        self._reporting_lock = RLock()
        self._pipeline_task_status_failed = None
        self._mock_execution = False  # used for nested pipelines (eager execution)
        self._last_progress_update_time = 0
        self._artifact_serialization_function = artifact_serialization_function
        self._artifact_deserialization_function = artifact_deserialization_function
        self._skip_global_imports = skip_global_imports
        self._enable_local_imports = enable_local_imports
        if not self._task:
            pipeline_project_args = self._create_pipeline_project_args(name, project)

            # if user disabled the auto-repo, we force local script storage (repo="" or repo=False)
            set_force_local_repo = False
            if Task.running_locally() and repo is not None and not repo:
                Task.force_store_standalone_script(force=True)
                set_force_local_repo = True

            self._task = Task.init(
                project_name=pipeline_project_args["project_name"],
                task_name=pipeline_project_args["task_name"],
                task_type=Task.TaskTypes.controller,
                auto_resource_monitoring=False,
                reuse_last_task_id=False,
            )

            # if user disabled the auto-repo, set it back to False (just in case)
            if set_force_local_repo:
                # noinspection PyProtectedMember
                self._task._wait_for_repo_detection(timeout=300.0)
                Task.force_store_standalone_script(force=False)

            self._create_pipeline_projects(
                task=self._task,
                parent_project=pipeline_project_args["parent_project"],
                project_name=pipeline_project_args["project_name"],
            )
            self._task.set_system_tags((self._task.get_system_tags() or []) + [self._tag])

        if output_uri is not None:
            self._task.output_uri = output_uri
        self._output_uri = output_uri
        self._task.set_base_docker(
            docker_image=docker,
            docker_arguments=docker_args,
            docker_setup_bash_script=docker_bash_setup_script,
        )
        self._task.set_packages(packages)
        self._task.set_script(
            repository=repo,
            branch=repo_branch,
            commit=repo_commit,
            working_dir=working_dir,
        )
        self._auto_connect_task = bool(self._task)
        # make sure we add to the main Task the pipeline tag
        if self._task and not self._pipeline_as_sub_project():
            self._task.add_tags([self._tag])

        self._monitored_nodes: Dict[str, dict] = {}
        self._abort_running_steps_on_failure = abort_on_failure
        self._def_max_retry_on_failure = retry_on_failure if isinstance(retry_on_failure, int) else 0
        self._retry_on_failure_callback = (
            retry_on_failure if callable(retry_on_failure) else self._default_retry_on_failure_callback
        )

        # add direct link to the pipeline page
        if self._pipeline_as_sub_project() and self._task:
            if add_run_number and self._task.running_locally():
                self._add_pipeline_name_run_number(self._task)
            # noinspection PyProtectedMember
            self._task.get_logger().report_text(
                "ClearML pipeline page: {}".format(
                    "{}/pipelines/{}/experiments/{}".format(
                        self._task._get_app_server(),
                        self._task.project if self._task.project is not None else "*",
                        self._task.id,
                    )
                )
            )

    @classmethod
    def _pipeline_as_sub_project(cls) -> bool:
        if cls._pipeline_as_sub_project_cached is None:
            cls._pipeline_as_sub_project_cached = bool(Session.check_min_api_server_version("2.17"))
        return cls._pipeline_as_sub_project_cached

    def set_default_execution_queue(self, default_execution_queue: Optional[str]) -> None:
        """
        Set the default execution queue if pipeline step does not specify an execution queue

        :param default_execution_queue: The execution queue to use if no execution queue is provided
        """
        self._default_execution_queue = str(default_execution_queue) if default_execution_queue else None

    def set_pipeline_execution_time_limit(self, max_execution_minutes: Optional[float]) -> None:
        """
        Set maximum execution time (minutes) for the entire pipeline. Pass None or 0 to disable execution time limit.

        :param float max_execution_minutes: The maximum time (minutes) for the entire pipeline process. The
            default is ``None``, indicating no time limit.
        """
        self._pipeline_time_limit = max_execution_minutes * 60.0 if max_execution_minutes else None

    def add_step(
        self,
        name: str,
        base_task_id: Optional[str] = None,
        parents: Optional[Sequence[str]] = None,
        parameter_override: Optional[Mapping[str, Any]] = None,
        configuration_overrides: Optional[Mapping[str, Union[str, Mapping]]] = None,
        task_overrides: Optional[Mapping[str, Any]] = None,
        execution_queue: Optional[str] = None,
        monitor_metrics: Optional[List[Union[Tuple[str, str], Tuple]]] = None,
        monitor_artifacts: Optional[List[Union[str, Tuple[str, str]]]] = None,
        monitor_models: Optional[List[Union[str, Tuple[str, str]]]] = None,
        time_limit: Optional[float] = None,
        base_task_project: Optional[str] = None,
        base_task_name: Optional[str] = None,
        clone_base_task: bool = True,
        continue_on_fail: bool = False,
        pre_execute_callback: Optional[
            Callable[["PipelineController", "PipelineController.Node", dict], bool]
        ] = None,  # noqa
        post_execute_callback: Optional[Callable[["PipelineController", "PipelineController.Node"], None]] = None,
        # noqa
        cache_executed_step: bool = False,
        base_task_factory: Optional[Callable[["PipelineController.Node"], Task]] = None,
        retry_on_failure: Optional[
            Union[
                int,
                Callable[["PipelineController", "PipelineController.Node", int], bool],
            ]
        ] = None,  # noqa
        status_change_callback: Optional[
            Callable[["PipelineController", "PipelineController.Node", str], None]
        ] = None,  # noqa
        recursively_parse_parameters: bool = False,
        output_uri: Optional[Union[str, bool]] = None,
        continue_behaviour: Optional[dict] = None,
        stage: Optional[str] = None
    ) -> bool:
        """
        Add a step to the pipeline execution DAG.
        Each step must have a unique name (this name will later be used to address the step)

        :param name: Unique of the step. For example `stage1`
        :param base_task_id: The Task ID to use for the step. Each time the step is executed,
            the base Task is cloned, then the cloned task will be sent for execution.
        :param parents: Optional list of parent nodes in the DAG.
            The current step in the pipeline will be sent for execution only after all the parent nodes
            have been executed successfully.
        :param parameter_override: Optional parameter overriding dictionary.
            The dict values can reference a previously executed step using the following form ``'${step_name}'``. Examples:

          - Artifact access ``parameter_override={'Args/input_file': '${<step_name>.artifacts.<artifact_name>.url}' }``
          - Model access (last model used) ``parameter_override={'Args/input_file': '${<step_name>.models.output.-1.url}' }``
          - Parameter access ``parameter_override={'Args/input_file': '${<step_name>.parameters.Args/input_file}' }``
          - Pipeline Task argument (see `Pipeline.add_parameter`) ``parameter_override={'Args/input_file': '${pipeline.<pipeline_parameter>}' }``
          - Task ID ``parameter_override={'Args/input_file': '${stage3.id}' }``
        :param recursively_parse_parameters: If True, recursively parse parameters from parameter_override in lists, dicts, or tuples.
            Example:

          - ``parameter_override={'Args/input_file': ['${<step_name>.artifacts.<artifact_name>.url}', 'file2.txt']}`` will be correctly parsed.
          - ``parameter_override={'Args/input_file': ('${<step_name_1>.parameters.Args/input_file}', '${<step_name_2>.parameters.Args/input_file}')}`` will be correctly parsed.
        :param configuration_overrides: Optional, override Task configuration objects.
            Expected dictionary of configuration object name and configuration object content.
            Examples:

          - ``{'General': dict(key='value')}``
          - ``{'General': 'configuration file content'}``
          - ``{'OmegaConf': YAML.dumps(full_hydra_dict)}``
        :param task_overrides: Optional task section overriding dictionary.
            The dict values can reference a previously executed step using the following form ``'${step_name}'``. Examples:

          - Get the latest commit from a specific branch ``task_overrides={'script.version_num': '', 'script.branch': 'main'}``
          - Match git repository branch to a previous step ``task_overrides={'script.branch': '${stage1.script.branch}', 'script.version_num': ''}``
          - Change container image ``task_overrides={'container.image': 'nvidia/cuda:11.6.0-devel-ubuntu20.04', 'container.arguments': '--ipc=host'}``
          - Match container image to a previous step ``task_overrides={'container.image': '${stage1.container.image}'}``
          - Reset requirements (the agent will use the "requirements.txt" inside the repo) ``task_overrides={'script.requirements.pip': ""}``
        :param execution_queue: Optional, the queue to use for executing this specific step.
            If not provided, the task will be sent to the default execution queue, as defined on the class
        :param monitor_metrics: Optional, log the step's metrics on the pipeline Task.
            Format is a list of pairs metric (title, series) to log: ``[(step_metric_title, step_metric_series), ]``.
            For example: ``[('test', 'accuracy'), ]``.
            Or a list of tuple pairs, to specify a different target metric for to use on the pipeline Task:
            ``[((step_metric_title, step_metric_series), (target_metric_title, target_metric_series)), ]``.
            For example: ``[[('test', 'accuracy'), ('model', 'accuracy')], ]``
        :param monitor_artifacts: Optional, log the step's artifacts on the pipeline Task.
            Provided a list of artifact names existing on the step's Task, they will also appear on the Pipeline itself.
            Example: ``[('processed_data', 'final_processed_data'), ]``.
            Alternatively user can also provide a list of artifacts to monitor
            (target artifact name will be the same as original artifact name).
            Example: ``['processed_data', ]``
        :param monitor_models: Optional, log the step's output models on the pipeline Task.
            Provided a list of model names existing on the step's Task, they will also appear on the Pipeline itself.
            Example: ``[('model_weights', 'final_model_weights'), ]``.
            Alternatively user can also provide a list of models to monitor
            (target models name will be the same as original model).
            Example: ``['model_weights', ]``.
            To select the latest (lexicographic) model use "model_*", or the last created model with just "*".
            Example:  ``['model_weights_*', ]``
        :param time_limit: Default None, no time limit.
            Step execution time limit, if exceeded the Task is aborted and the pipeline is stopped and marked failed.
        :param base_task_project: If base_task_id is not given,
            use the base_task_project and base_task_name combination to retrieve the base_task_id to use for the step.
        :param base_task_name: If base_task_id is not given,
            use the base_task_project and base_task_name combination to retrieve the base_task_id to use for the step.
        :param clone_base_task: If True (default), the pipeline will clone the base task, and modify/enqueue
            the cloned Task. If False, the base-task is used directly, notice it has to be in draft-mode (created).
        :param continue_on_fail: (Deprecated, use `continue_behaviour` instead).
            If True, failed step will not cause the pipeline to stop
            (or marked as failed). Notice, that steps that are connected (or indirectly connected)
            to the failed step will be skipped. Defaults to False
        :param pre_execute_callback: Callback function, called when the step (Task) is created
            and before it is sent for execution. Allows a user to modify the Task before launch.
            Use `node.job` to access the ClearmlJob object, or `node.job.task` to directly access the Task object.
            ``parameters`` are the configuration arguments passed to the ClearmlJob.

            If the callback returned value is `False`,
            the Node is skipped and so is any node in the DAG that relies on this node.

            Notice the `parameters` are already parsed,
            e.g. ``${step1.parameters.Args/param}`` is replaced with relevant value.

            .. code-block:: py

                def step_created_callback(
                    pipeline,             # type: PipelineController,
                    node,                 # type: PipelineController.Node,
                    parameters,           # type: dict
                ):
                    pass

        :param post_execute_callback: Callback function, called when a step (Task) is completed
            and other jobs are executed. Allows a user to modify the Task status after completion.

            .. code-block:: py

                def step_completed_callback(
                    pipeline,             # type: PipelineController,
                    node,                 # type: PipelineController.Node,
                ):
                    pass

        :param cache_executed_step: If True, before launching the new step,
            after updating with the latest configuration, check if an exact Task with the same parameter/code
            was already executed. If it was found, use it instead of launching a new Task.
            Default: False, a new cloned copy of base_task is always used.
            Notice: If the git repo reference does not have a specific commit ID, the Task will never be used.
            If `clone_base_task` is False there is no cloning, hence the base_task is used.
        :param base_task_factory: Optional, instead of providing a pre-existing Task,
            provide a Callable function to create the Task (returns Task object)
        :param retry_on_failure: Integer (number of retries) or Callback function that returns True to allow a retry

          - Integer: In case of node failure, retry the node the number of times indicated by this parameter.
          - Callable: A function called on node failure. Takes as parameters:
            the PipelineController instance, the PipelineController.Node that failed and an int
            representing the number of previous retries for the node that failed.
            The function must return ``True`` if the node should be retried and ``False`` otherwise.
            If True, the node will be re-queued and the number of retries left will be decremented by 1.
            By default, if this callback is not specified, the function will be retried the number of
            times indicated by `retry_on_failure`.

              .. code-block:: py

                  def example_retry_on_failure_callback(pipeline, node, retries):
                      print(node.name, ' failed')
                      # allow up to 5 retries (total of 6 runs)
                      return retries < 5

        :param status_change_callback: Callback function, called when the status of a step (Task) changes.
            Use `node.job` to access the ClearmlJob object, or `node.job.task` to directly access the Task object.
            The signature of the function must look the following way:

            .. code-block:: py

                def status_change_callback(
                    pipeline,             # type: PipelineController,
                    node,                 # type: PipelineController.Node,
                    previous_status       # type: str
                ):
                    pass

        :param output_uri: The storage / output url for this step. This is the default location for output
            models and other artifacts. Check Task.init reference docs for more info (output_uri is a parameter).
        :param continue_behaviour: Controls whether the pipeline will continue running after a step failed/was aborted.
            Different behaviours can be set using a dictionary of boolean options. Supported options are:

          - continue_on_fail - If True, the pipeline will continue even if the step failed.
            If False, the pipeline will stop
          - continue_on_abort - If True, the pipeline will continue even if the step was aborted.
            If False, the pipeline will stop
          - skip_children_on_fail - If True, the children of this step will be skipped if it failed.
            If False, the children will run even if this step failed.
            Any parameters passed from the failed step to its children will default to None
          - skip_children_on_abort - If True, the children of this step will be skipped if it was aborted.
            If False, the children will run even if this step was aborted.
            Any parameters passed from the failed step to its children will default to None
          - If the keys are not present in the dictionary, their values will default to True
        :param stage: Name of the stage. This parameter enables pipeline step grouping into stages

        :return: True if successful
        """
        if continue_on_fail:
            warnings.warn(
                "`continue_on_fail` is deprecated. Use `continue_behaviour` instead",
                DeprecationWarning,
            )
        # always store callback functions (even when running remotely)
        if pre_execute_callback:
            self._pre_step_callbacks[name] = pre_execute_callback
        if post_execute_callback:
            self._post_step_callbacks[name] = post_execute_callback

        self._verify_node_name(name)

        if not base_task_factory and not base_task_id:
            if not base_task_project or not base_task_name:
                raise ValueError("Either base_task_id or base_task_project/base_task_name must be provided")
            base_task = Task.get_task(
                project_name=base_task_project,
                task_name=base_task_name,
                allow_archived=True,
                task_filter=dict(
                    status=[
                        str(Task.TaskStatusEnum.created),
                        str(Task.TaskStatusEnum.queued),
                        str(Task.TaskStatusEnum.in_progress),
                        str(Task.TaskStatusEnum.published),
                        str(Task.TaskStatusEnum.stopped),
                        str(Task.TaskStatusEnum.completed),
                        str(Task.TaskStatusEnum.closed),
                    ],
                ),
            )
            if not base_task:
                raise ValueError(
                    "Could not find base_task_project={} base_task_name={}".format(base_task_project, base_task_name)
                )
            if Task.archived_tag in base_task.get_system_tags():
                LoggerRoot.get_base_logger().warning(
                    "Found base_task_project={} base_task_name={} but it is archived".format(
                        base_task_project, base_task_name
                    )
                )
            base_task_id = base_task.id

        if configuration_overrides is not None:
            # verify we have a dict or a string on all values
            if not isinstance(configuration_overrides, dict) or not all(
                isinstance(v, (str, dict)) for v in configuration_overrides.values()
            ):
                raise ValueError(
                    "configuration_overrides must be a dictionary, with all values "
                    "either dicts or strings, got '{}' instead".format(configuration_overrides)
                )

        if task_overrides:
            task_overrides = flatten_dictionary(task_overrides, sep=".")

        self._nodes[name] = self.Node(
            name=name,
            base_task_id=base_task_id,
            parents=parents or [],
            queue=execution_queue,
            timeout=time_limit,
            parameters=parameter_override or {},
            recursively_parse_parameters=recursively_parse_parameters,
            configurations=configuration_overrides,
            clone_task=clone_base_task,
            task_overrides=task_overrides,
            cache_executed_step=cache_executed_step,
            continue_on_fail=continue_on_fail,
            task_factory_func=base_task_factory,
            monitor_metrics=monitor_metrics or [],
            monitor_artifacts=monitor_artifacts or [],
            monitor_models=monitor_models or [],
            output_uri=self._output_uri if output_uri is None else output_uri,
            continue_behaviour=continue_behaviour,
            stage=stage
        )
        self._retries[name] = 0
        self._retries_callbacks[name] = (
            retry_on_failure
            if callable(retry_on_failure)
            else (
                functools.partial(
                    self._default_retry_on_failure_callback,
                    max_retries=retry_on_failure,
                )
                if isinstance(retry_on_failure, int)
                else self._retry_on_failure_callback
            )
        )
        if status_change_callback:
            self._status_change_callbacks[name] = status_change_callback

        if self._task and not self._task.running_locally():
            self.update_execution_plot()

        return True

    def add_function_step(
        self,
        name: str,
        function: Callable,
        function_kwargs: Optional[Dict[str, Any]] = None,
        function_return: Optional[List[str]] = None,
        project_name: Optional[str] = None,
        task_name: Optional[str] = None,
        task_type: Optional[str] = None,
        auto_connect_frameworks: Optional[dict] = None,
        auto_connect_arg_parser: Optional[dict] = None,
        packages: Optional[Union[bool, str, Sequence[str]]] = None,
        repo: Optional[str] = None,
        repo_branch: Optional[str] = None,
        repo_commit: Optional[str] = None,
        helper_functions: Optional[Sequence[Callable]] = None,
        docker: Optional[str] = None,
        docker_args: Optional[str] = None,
        docker_bash_setup_script: Optional[str] = None,
        parents: Optional[Sequence[str]] = None,
        execution_queue: Optional[str] = None,
        monitor_metrics: Optional[List[Tuple]] = None,
        monitor_artifacts: Optional[List[Union[str, Tuple]]] = None,
        monitor_models: Optional[List[Union[str, Tuple]]] = None,
        time_limit: Optional[float] = None,
        continue_on_fail: bool = False,
        pre_execute_callback: Optional[
            Callable[["PipelineController", "PipelineController.Node", dict], bool]
        ] = None,  # noqa
        post_execute_callback: Optional[Callable[["PipelineController", "PipelineController.Node"], None]] = None,
        # noqa
        cache_executed_step: bool = False,
        retry_on_failure: Optional[
            Union[
                int,
                Callable[["PipelineController", "PipelineController.Node", int], bool],
            ]
        ] = None,  # noqa
        status_change_callback: Optional[
            Callable[["PipelineController", "PipelineController.Node", str], None]
        ] = None,  # noqa
        tags: Optional[Union[str, Sequence[str]]] = None,
        output_uri: Optional[Union[str, bool]] = None,
        draft: Optional[bool] = False,
        working_dir: Optional[str] = None,
        continue_behaviour: Optional[dict] = None,
        stage: Optional[str] = None
    ) -> bool:
        """
        Create a Task from a function, including wrapping the function input arguments
        into the hyper-parameter section as kwargs, and storing function results as named artifacts

        Example:

        .. code-block:: py

            def mock_func(a=6, b=9):
                c = a*b
                print(a, b, c)
                return c, c**2

            create_task_from_function(mock_func, function_return=['mul', 'square'])

        Example arguments from other Tasks (artifact):

        .. code-block:: py

            def mock_func(matrix_np):
                c = matrix_np*matrix_np
                print(matrix_np, c)
                return c

            create_task_from_function(
                mock_func,
                function_kwargs={'matrix_np': 'aabb1122.previous_matrix'},
                function_return=['square_matrix']
            )

        :param name: Unique of the step. For example `stage1`
        :param function: A global function to convert into a standalone Task
        :param function_kwargs: Optional, provide subset of function arguments and default values to expose.
            If not provided automatically take all function arguments & defaults
            Optional, pass input arguments to the function from other Tasks' output artifact.
            Example argument named `numpy_matrix` from Task ID `aabbcc` artifact name `answer`:
            ``{'numpy_matrix': 'aabbcc.answer'}``
        :param function_return: Provide a list of names for all the results.
            If not provided, no results will be stored as artifacts.
        :param project_name: Set the project name for the task. Required if base_task_id is None.
        :param task_name: Set the name of the remote task, if not provided use `name` argument.
        :param task_type: Optional, The task type to be created. Supported values: 'training', 'testing', 'inference',
            'data_processing', 'application', 'monitor', 'controller', 'optimizer', 'service', 'qc', 'custom'
        :param auto_connect_frameworks: Control the frameworks auto connect, see `Task.init` auto_connect_frameworks
        :param auto_connect_arg_parser: Control the ArgParser auto connect, see `Task.init` auto_connect_arg_parser
        :param packages: Manually specify a list of required packages or a local requirements.txt file.
            Example: ["tqdm>=2.1", "scikit-learn"] or "./requirements.txt"
            If not provided, packages are automatically added based on the imports used in the function.
            Use `False` to install requirements from "requirements.txt" inside your git repository
        :param repo: Optional, specify a repository to attach to the function, when remotely executing.
            Allow users to execute the function inside the specified repository, enabling to load modules/script
            from a repository Notice the execution work directory will be the repository root folder.
            Supports both git repo url link, and local repository path.
            Example remote url: 'https://github.com/user/repo.git'
            Example local repo copy: './repo' -> will automatically store the remote
            repo url and commit ID based on the locally cloned copy
        :param repo_branch: Optional, specify the remote repository branch (Ignored, if local repo path is used)
        :param repo_commit: Optional, specify the repository commit ID (Ignored, if local repo path is used)
        :param helper_functions: Optional, a list of helper functions to make available
            for the standalone function Task.
        :param docker: Select the docker image to be executed in by the remote session
        :param docker_args: Add docker arguments, pass a single string
        :param docker_bash_setup_script: Add bash script to be executed
            inside the docker before setting up the Task's environment
        :param parents: Optional list of parent nodes in the DAG.
            The current step in the pipeline will be sent for execution only after all the parent nodes
            have been executed successfully.
        :param execution_queue: Optional, the queue to use for executing this specific step.
            If not provided, the task will be sent to the default execution queue, as defined on the class
        :param monitor_metrics: Optional, log the step's metrics on the pipeline Task.
            Format is a list of pairs metric (title, series) to log: ``[(step_metric_title, step_metric_series), ]``.
            For example: ``[('test', 'accuracy'), ]``.
            Or a list of tuple pairs, to specify a different target metric for to use on the pipeline Task:
            ``[((step_metric_title, step_metric_series), (target_metric_title, target_metric_series)), ]``.
            For example: ``[[('test', 'accuracy'), ('model', 'accuracy')], ]``
        :param monitor_artifacts: Optional, log the step's artifacts on the pipeline Task.
            Provided a list of artifact names existing on the step's Task, they will also appear on the Pipeline itself.
            Example: ``[('processed_data', 'final_processed_data'), ]``.
            Alternatively user can also provide a list of artifacts to monitor
            (target artifact name will be the same as original artifact name).
            Example: ``['processed_data', ]``
        :param monitor_models: Optional, log the step's output models on the pipeline Task.
            Provided a list of model names existing on the step's Task, they will also appear on the Pipeline itself.
            Example: ``[('model_weights', 'final_model_weights'), ]``.
            Alternatively user can also provide a list of models to monitor
            (target models name will be the same as original model).
            Example: ``['model_weights', ]``.
            To select the latest (lexicographic) model use "model_*", or the last created model with just "*".
            Example:  ``['model_weights_*', ]``
        :param time_limit: Default None, no time limit.
            Step execution time limit, if exceeded the Task is aborted and the pipeline is stopped and marked failed.
        :param continue_on_fail: (Deprecated, use `continue_behaviour` instead).
            If True, failed step will not cause the pipeline to stop
            (or marked as failed). Notice, that steps that are connected (or indirectly connected)
            to the failed step will be skipped. Defaults to False
        :param pre_execute_callback: Callback function, called when the step (Task) is created
            and before it is sent for execution. Allows a user to modify the Task before launch.
            Use `node.job` to access the ClearmlJob object, or `node.job.task` to directly access the Task object.
            ``parameters`` are the configuration arguments passed to the ClearmlJob.

            If the callback returned value is `False`,
            the Node is skipped and so is any node in the DAG that relies on this node.

            Notice the `parameters` are already parsed,
            e.g. ``${step1.parameters.Args/param}`` is replaced with relevant value.

            .. code-block:: py

                def step_created_callback(
                    pipeline,             # type: PipelineController,
                    node,                 # type: PipelineController.Node,
                    parameters,           # type: dict
                ):
                    pass

        :param post_execute_callback: Callback function, called when a step (Task) is completed
            and other jobs are executed. Allows a user to modify the Task status after completion.

            .. code-block:: py

                def step_completed_callback(
                    pipeline,             # type: PipelineController,
                    node,                 # type: PipelineController.Node,
                ):
                    pass

        :param cache_executed_step: If True, before launching the new step,
            after updating with the latest configuration, check if an exact Task with the same parameter/code
            was already executed. If it was found, use it instead of launching a new Task.
            Default: False, a new cloned copy of base_task is always used.
            Notice: If the git repo reference does not have a specific commit ID, the Task will never be used.
        :param retry_on_failure: Integer (number of retries) or Callback function that returns True to allow a retry

          - Integer: In case of node failure, retry the node the number of times indicated by this parameter.
          - Callable: A function called on node failure. Takes as parameters:
            the PipelineController instance, the PipelineController.Node that failed and an int
            representing the number of previous retries for the node that failed.
            The function must return ``True`` if the node should be retried and ``False`` otherwise.
            If True, the node will be re-queued and the number of retries left will be decremented by 1.
            By default, if this callback is not specified, the function will be retried the number of
            times indicated by `retry_on_failure`.

              .. code-block:: py

                  def example_retry_on_failure_callback(pipeline, node, retries):
                      print(node.name, ' failed')
                      # allow up to 5 retries (total of 6 runs)
                      return retries < 5

        :param status_change_callback: Callback function, called when the status of a step (Task) changes.
            Use `node.job` to access the ClearmlJob object, or `node.job.task` to directly access the Task object.
            The signature of the function must look the following way:

            .. code-block:: py

                def status_change_callback(
                    pipeline,             # type: PipelineController,
                    node,                 # type: PipelineController.Node,
                    previous_status       # type: str
                ):
                    pass

        :param tags: A list of tags for the specific pipeline step.
            When executing a Pipeline remotely
            (i.e. launching the pipeline from the UI/enqueuing it), this method has no effect.
        :param output_uri: The storage / output url for this step. This is the default location for output
            models and other artifacts. Check Task.init reference docs for more info (output_uri is a parameter).
        :param draft: (default False). If True, the Task will be created as a draft task.
        :param working_dir: Working directory to launch the script from.
        :param continue_behaviour: Controls whether the pipeline will continue running after a step failed/was aborted.
            Different behaviours can be set using a dictionary of boolean options. Supported options are:

          - continue_on_fail - If True, the pipeline will continue even if the step failed.
            If False, the pipeline will stop
          - continue_on_abort - If True, the pipeline will continue even if the step was aborted.
            If False, the pipeline will stop
          - skip_children_on_fail - If True, the children of this step will be skipped if it failed.
            If False, the children will run even if this step failed. Any parameters passed from the failed step to its
            children will default to None
          - skip_children_on_abort - If True, the children of this step will be skipped if it was aborted.
            If False, the children will run even if this step was aborted.
            Any parameters passed from the failed step to its children will default to None
          - If the keys are not present in the dictionary, their values will default to True
        :param stage: Name of the stage. This parameter enables pipeline step grouping into stages

        :return: True if successful
        """
        if continue_on_fail:
            warnings.warn(
                "`continue_on_fail` is deprecated. Use `continue_behaviour` instead",
                DeprecationWarning,
            )

        function_kwargs = function_kwargs or {}
        default_kwargs = inspect.getfullargspec(function)
        if default_kwargs and default_kwargs.args and default_kwargs.defaults:
            for key, val in zip(
                default_kwargs.args[-len(default_kwargs.defaults) :],
                default_kwargs.defaults,
            ):
                function_kwargs.setdefault(key, val)

        return self._add_function_step(
            name=name,
            function=function,
            function_kwargs=function_kwargs,
            function_return=function_return,
            project_name=project_name,
            task_name=task_name,
            task_type=task_type,
            auto_connect_frameworks=auto_connect_frameworks,
            auto_connect_arg_parser=auto_connect_arg_parser,
            packages=packages,
            repo=repo,
            repo_branch=repo_branch,
            repo_commit=repo_commit,
            helper_functions=helper_functions,
            docker=docker,
            docker_args=docker_args,
            docker_bash_setup_script=docker_bash_setup_script,
            parents=parents,
            execution_queue=execution_queue,
            monitor_metrics=monitor_metrics,
            monitor_artifacts=monitor_artifacts,
            monitor_models=monitor_models,
            time_limit=time_limit,
            continue_on_fail=continue_on_fail,
            pre_execute_callback=pre_execute_callback,
            post_execute_callback=post_execute_callback,
            cache_executed_step=cache_executed_step,
            retry_on_failure=retry_on_failure,
            status_change_callback=status_change_callback,
            tags=tags,
            output_uri=output_uri,
            draft=draft,
            working_dir=working_dir,
            continue_behaviour=continue_behaviour,
            stage=stage
        )

    def start(
        self,
        queue: str = "services",
        step_task_created_callback: Optional[
            Callable[["PipelineController", "PipelineController.Node", dict], bool]
        ] = None,  # noqa
        step_task_completed_callback: Optional[
            Callable[["PipelineController", "PipelineController.Node"], None]
        ] = None,  # noqa
        wait: bool = True,
    ) -> bool:
        """
        Start the current pipeline remotely (on the selected services queue).
        The current process will be stopped and launched remotely.

        :param queue: queue name to launch the pipeline on
        :param Callable step_task_created_callback: Callback function, called when a step (Task) is created
            and before it is sent for execution. Allows a user to modify the Task before launch.
            Use `node.job` to access the ClearmlJob object, or `node.job.task` to directly access the Task object.
            `parameters` are the configuration arguments passed to the ClearmlJob.

            If the callback returned value is `False`,
            the Node is skipped and so is any node in the DAG that relies on this node.

            Notice the `parameters` are already parsed,
            e.g. ``${step1.parameters.Args/param}`` is replaced with relevant value.

            .. code-block:: py

                def step_created_callback(
                    pipeline,             # type: PipelineController,
                    node,                 # type: PipelineController.Node,
                    parameters,           # type: dict
                ):
                    pass

        :param Callable step_task_completed_callback: Callback function, called when a step (Task) is completed
            and other jobs are executed. Allows a user to modify the Task status after completion.

            .. code-block:: py

                def step_completed_callback(
                    pipeline,             # type: PipelineController,
                    node,                 # type: PipelineController.Node,
                ):
                    pass
        :param wait: If True (default), start the pipeline controller, return only
            after the pipeline is done (completed/aborted/failed)

        :return: True, if the controller started. False, if the controller did not start.

        """
        if not self._task:
            raise ValueError(
                "Could not find main Task, PipelineController must be created with `always_create_task=True`"
            )

        # serialize state only if we are running locally
        if Task.running_locally() or not self._task.is_main_task():
            self._verify()
            self._serialize_pipeline_task()
            self.update_execution_plot()

        # stop current Task and execute remotely or no-op
        self._task.execute_remotely(queue_name=queue, exit_process=True, clone=False)

        if not Task.running_locally() and self._task.is_main_task():
            self._start(
                step_task_created_callback=step_task_created_callback,
                step_task_completed_callback=step_task_completed_callback,
                wait=wait,
            )

        return True

    def start_locally(self, run_pipeline_steps_locally: bool = False) -> None:
        """
        Start the current pipeline locally, meaning the pipeline logic is running on the current machine,
        instead of on the `services` queue.

        Using run_pipeline_steps_locally=True you can run all the pipeline steps locally as sub-processes.
        Notice: when running pipeline steps locally, it assumes local code execution
        (i.e. it is running the local code as is, regardless of the git commit/diff on the pipeline steps Task)

        :param run_pipeline_steps_locally: (default False) If True, run the pipeline steps themselves locally as a
            subprocess (use for debugging the pipeline locally, notice the pipeline code is expected to be available
            on the local machine)
        """
        if not self._task:
            raise ValueError(
                "Could not find main Task, PipelineController must be created with `always_create_task=True`"
            )

        if run_pipeline_steps_locally:
            self._clearml_job_class = LocalClearmlJob
            self._default_execution_queue = self._default_execution_queue or "mock"

        # serialize state only if we are running locally
        if Task.running_locally() or not self._task.is_main_task():
            self._verify()
            self._serialize_pipeline_task()

        self._start(wait=True)

    def create_draft(self) -> None:
        """
        Optional, manually create & serialize the Pipeline Task (use with care for manual multi pipeline creation).

        **Notice** The recommended flow would be to call `pipeline.start(queue=None)`
        which would have a similar effect and will allow you to clone/enqueue later on.

        After calling Pipeline.create(), users can edit the pipeline in the UI and enqueue it for execution.

        Notice: this function should be used to programmatically create pipeline for later usage.
        To automatically create and launch pipelines, call the `start()` method.
        """
        self._verify()
        self._serialize_pipeline_task()
        self._task.close()
        self._task.reset()

    def connect_configuration(
        self,
        configuration: Union[Mapping, list, Path, str],
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Union[dict, Path, str]:
        """
        Connect a configuration dictionary or configuration file (pathlib.Path / str) to the PipelineController object.
        This method should be called before reading the configuration file.

        For example, a local file:

        .. code-block:: py

           config_file = pipe.connect_configuration(config_file)
           my_params = json.load(open(config_file,'rt'))

        A parameter dictionary/list:

        .. code-block:: py

           my_params = pipe.connect_configuration(my_params)

        :param configuration: The configuration. This is usually the configuration used in the model training process.
            Specify one of the following:

          - A dictionary/list - A dictionary containing the configuration. ClearML stores the configuration in
            the **ClearML Server** (backend), in a HOCON format (JSON-like format) which is editable.
          - A ``pathlib2.Path`` string - A path to the configuration file. ClearML stores the content of the file.
            A local path must be relative path. When executing a pipeline remotely in a worker, the contents brought
            from the **ClearML Server** (backend) overwrites the contents of the file.

        :param str name: Configuration section name. default: 'General'
            Allowing users to store multiple configuration dicts/files

        :param str description: Configuration section description (text). default: None

        :return: If a dictionary is specified, then a dictionary is returned. If pathlib2.Path / string is
            specified, then a path to a local configuration file is returned. Configuration object.
        """
        return self._task.connect_configuration(configuration, name=name, description=description)

    @classmethod
    def get_logger(cls) -> Logger:
        """
        Return a logger connected to the Pipeline Task.
        The logger can be used by any function/tasks executed by the pipeline, in order to report
        directly to the pipeline Task itself. It can also be called from the main pipeline control Task.

        Raise ValueError if main Pipeline task could not be located.

        :return: Logger object for reporting metrics (scalars, plots, debug samples etc.)
        """
        return cls._get_pipeline_task().get_logger()

    @classmethod
    def upload_model(
        cls,
        model_name: str,
        model_local_path: str,
        upload_uri: Optional[str] = None,
    ) -> OutputModel:
        """
        Upload (add) a model to the main Pipeline Task object.
        This function can be called from any pipeline component to directly add models into the main pipeline Task

        The model file/path will be uploaded to the Pipeline Task and registered on the model repository.

        Raise ValueError if main Pipeline task could not be located.

        :param model_name: Model name as will appear in the model registry (in the pipeline's project)
        :param model_local_path: Path to the local model file or directory to be uploaded.
            If a local directory is provided the content of the folder (recursively) will be
            packaged into a zip file and uploaded
        :param upload_uri: The URI of the storage destination for model weights upload. The default value
            is the previously used URI.

        :return: The uploaded OutputModel
        """
        task = cls._get_pipeline_task()
        model_name = str(model_name)
        model_local_path = Path(model_local_path)
        out_model = OutputModel(task=task, name=model_name)
        out_model.update_weights(weights_filename=model_local_path.as_posix(), upload_uri=upload_uri)
        return out_model

    @classmethod
    def upload_artifact(
        cls,
        name: str,
        artifact_object: Any,
        metadata: Optional[Mapping] = None,
        delete_after_upload: bool = False,
        auto_pickle: Optional[bool] = None,
        preview: Any = None,
        wait_on_upload: bool = False,
        serialization_function: Optional[Callable[[Any], Union[bytes, bytearray]]] = None,
        sort_keys: bool = True,
    ) -> bool:
        """
        Upload (add) an artifact to the main Pipeline Task object.
        This function can be called from any pipeline component to directly add artifacts into the main pipeline Task.

        The artifact can be uploaded by any function/tasks executed by the pipeline, in order to report
        directly to the pipeline Task itself. It can also be called from the main pipeline control Task.

        Raise ValueError if main Pipeline task could not be located.

        The currently supported upload artifact types include:
        - string / Path - A path to artifact file. If a wildcard or a folder is specified, then ClearML
        creates and uploads a ZIP file.
        - dict - ClearML stores a dictionary as ``.json`` file and uploads it.
        - pandas.DataFrame - ClearML stores a pandas.DataFrame as ``.csv.gz`` (compressed CSV) file and uploads it.
        - numpy.ndarray - ClearML stores a numpy.ndarray as ``.npz`` file and uploads it.
        - PIL.Image - ClearML stores a PIL.Image as ``.png`` file and uploads it.
        - Any - If called with auto_pickle=True, the object will be pickled and uploaded.

        :param name: The artifact name.

            .. warning::
               If an artifact with the same name was previously uploaded, then it is overwritten.

        :param artifact_object:  The artifact object.
        :param metadata: A dictionary of key-value pairs for any metadata. This dictionary appears with the
            experiment in the **ClearML Web-App (UI)**, **ARTIFACTS** tab.
        :param delete_after_upload: After the upload, delete the local copy of the artifact

            - ``True`` - Delete the local copy of the artifact.
            - ``False`` - Do not delete. (default)

        :param auto_pickle: If True, and the artifact_object is not one of the following types:
            pathlib2.Path, dict, pandas.DataFrame, numpy.ndarray, PIL.Image, url (string), local_file (string)
            the artifact_object will be pickled and uploaded as pickle file artifact (with file extension .pkl)
            If set to None (default) the sdk.development.artifacts.auto_pickle configuration value will be used.

        :param preview: The artifact preview

        :param wait_on_upload: Whether the upload should be synchronous, forcing the upload to complete
            before continuing.

        :param serialization_function: A serialization function that takes one
            parameter of any type which is the object to be serialized. The function should return
            a `bytes` or `bytearray` object, which represents the serialized object. Note that the object will be
            immediately serialized using this function, thus other serialization methods will not be used
            (e.g. `pandas.DataFrame.to_csv`), even if possible. To deserialize this artifact when getting
            it using the `Artifact.get` method, use its `deserialization_function` argument.

        :param sort_keys: If True (default), sort the keys of the artifact if it is yaml/json serializable.
            Otherwise, don't sort the keys. Ignored if the artifact is not yaml/json serializable.

        :return: The status of the upload.

          - ``True`` - Upload succeeded.
          - ``False`` - Upload failed.

        :raise: If the artifact object type is not supported, raise a ``ValueError``.
        """
        task = cls._get_pipeline_task()
        return task.upload_artifact(
            name=name,
            artifact_object=artifact_object,
            metadata=metadata,
            delete_after_upload=delete_after_upload,
            auto_pickle=auto_pickle,
            preview=preview,
            wait_on_upload=wait_on_upload,
            serialization_function=serialization_function,
            sort_keys=sort_keys,
        )

    def stop(
        self,
        timeout: Optional[float] = None,
        mark_failed: bool = False,
        mark_aborted: bool = False,
    ) -> ():
        """
        Stop the pipeline controller and the optimization thread.
        If mark_failed and mark_aborted are False (default) mark the pipeline as completed,
        unless one of the steps failed, then mark the pipeline as failed.

        :param timeout: Wait timeout for the optimization thread to exit (minutes).
            The default is ``None``, indicating do not wait to terminate immediately.
        :param mark_failed: If True, mark the pipeline task as failed. (default False)
        :param mark_aborted: If True, mark the pipeline task as aborted. (default False)
        """
        self._stop_event.set()

        self.wait(timeout=timeout)
        if not self._task:
            return

        # sync pipeline state
        self.update_execution_plot()

        self._task.close()
        if mark_failed:
            self._task.mark_failed(status_reason="Pipeline aborted and failed", force=True)
        elif mark_aborted:
            self._task.mark_stopped(status_message="Pipeline aborted", force=True)
        elif self._pipeline_task_status_failed:
            print("Setting pipeline controller Task as failed (due to failed steps) !")
            self._task.mark_failed(status_reason="Pipeline step failed", force=True)

    def wait(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for the pipeline to finish.

        .. note::
            This method does not stop the pipeline. Call :meth:`stop` to terminate the pipeline.

        :param float timeout: The timeout to wait for the pipeline to complete (minutes).
            If ``None``, then wait until we reached the timeout, or pipeline completed.

        :return: True, if the pipeline finished. False, if the pipeline timed out.

        """
        if not self.is_running():
            return True

        if timeout is not None:
            timeout *= 60.0

        _thread = self._thread

        _thread.join(timeout=timeout)
        if _thread.is_alive():
            return False

        return True

    def is_running(self) -> bool:
        """
        return True if the pipeline controller is running.

        :return: A boolean indicating whether the pipeline controller is active (still running) or stopped.
        """
        return self._thread is not None and self._thread.is_alive()

    def is_successful(self, fail_on_step_fail: bool = True, fail_condition: str = "all") -> bool:
        """
        Evaluate whether the pipeline is successful.

        :param fail_on_step_fail: If True (default), evaluate the pipeline steps' status to assess if the pipeline
            is successful. If False, only evaluate the controller
        :param fail_condition: Must be one of the following: 'all' (default), 'failed' or 'aborted'. If 'failed', this
            function will return False if the pipeline failed and True if the pipeline was aborted. If 'aborted',
            this function will return False if the pipeline was aborted and True if the pipeline failed. If 'all',
            this function will return False in both cases.

        :return: A boolean indicating whether the pipeline was successful or not. Note that if the pipeline is in a
            running/pending state, this function will return False
        """
        if fail_condition == "all":
            success_status = [Task.TaskStatusEnum.completed]
        elif fail_condition == "failed":
            success_status = [
                Task.TaskStatusEnum.completed,
                Task.TaskStatusEnum.stopped,
            ]
        elif fail_condition == "aborted":
            success_status = [Task.TaskStatusEnum.completed, Task.TaskStatusEnum.failed]
        else:
            raise UsageError("fail_condition needs to be one of the following: 'all', 'failed', 'aborted'")
        if self._task.status not in success_status:
            return False
        if not fail_on_step_fail:
            return True
        self._update_nodes_status()
        for node in self._nodes.values():
            if node.status not in success_status:
                return False
        return True

    def elapsed(self) -> float:
        """
        Return minutes elapsed from controller stating time stamp.

        :return: The minutes from controller start time. A negative value means the process has not started yet.
        """
        if self._start_time is None:
            return -1.0
        return (time() - self._start_time) / 60.0

    def get_pipeline_dag(self) -> Mapping[str, "PipelineController.Node"]:
        """
        Return the pipeline execution graph, each node in the DAG is PipelineController.Node object.
        Graph itself is a dictionary of Nodes (key based on the Node name),
        each node holds links to its parent Nodes (identified by their unique names)

        :return: execution tree, as a nested dictionary. Example:

        .. code-block:: py

            {
                'stage1' : Node() {
                    name: 'stage1'
                    job: ClearmlJob
                    ...
                },
            }

        """
        return self._nodes

    def get_processed_nodes(self) -> Sequence["PipelineController.Node"]:
        """
        Return a list of the processed pipeline nodes, each entry in the list is PipelineController.Node object.

        :return: executed (excluding currently executing) nodes list
        """
        return {k: n for k, n in self._nodes.items() if n.executed}

    def get_running_nodes(self) -> Sequence["PipelineController.Node"]:
        """
        Return a list of the currently running pipeline nodes,
        each entry in the list is PipelineController.Node object.

        :return: Currently running nodes list
        """
        return {k: n for k, n in self._nodes.items() if k in self._running_nodes}

    def update_execution_plot(self) -> ():
        """
        Update sankey diagram of the current pipeline
        """
        with self._reporting_lock:
            self._update_execution_plot()
        # also trigger node monitor scanning
        self._scan_monitored_nodes()

    def add_parameter(
        self,
        name: str,
        default: Optional[Any] = None,
        description: Optional[str] = None,
        param_type: Optional[str] = None,
    ) -> None:
        """
        Add a parameter to the pipeline Task.
        The parameter can be used as input parameter for any step in the pipeline.
        Notice all parameters will appear under the PipelineController Task's Hyper-parameters -> Pipeline section
        Example: pipeline.add_parameter(name='dataset', description='dataset ID to process the pipeline')
        Then in one of the steps we can refer to the value of the parameter with ``'${pipeline.dataset}'``

        :param name: String name of the parameter.
        :param default: Default value to be put as the default value (can be later changed in the UI)
        :param description: String description of the parameter and its usage in the pipeline
        :param param_type: Optional, parameter type information (to be used as hint for casting and description)
        """
        self._pipeline_args[str(name)] = default
        if description:
            self._pipeline_args_desc[str(name)] = str(description)
        if param_type:
            self._pipeline_args_type[str(name)] = param_type

    def get_parameters(self) -> dict:
        """
        Return the pipeline parameters dictionary

        :return: Dictionary str -> str
        """
        return self._pipeline_args

    @classmethod
    def _create_pipeline_project_args(cls, name: str, project: str) -> dict:
        task_name = name or project or "{}".format(datetime.now())
        if cls._pipeline_as_sub_project():
            parent_project = (project + "/" if project else "") + cls._project_section
            project_name = "{}/{}".format(parent_project, task_name)
        else:
            parent_project = None
            project_name = project or "Pipelines"
        return {
            "task_name": task_name,
            "parent_project": parent_project,
            "project_name": project_name,
        }

    @classmethod
    def _create_pipeline_projects(cls, task: Task, parent_project: str, project_name: str) -> None:
        # make sure project is hidden
        if not cls._pipeline_as_sub_project():
            return
        get_or_create_project(
            Task._get_default_session(),
            project_name=parent_project,
            system_tags=["hidden"],
        )
        return get_or_create_project(
            Task._get_default_session(),
            project_name=project_name,
            project_id=task.project,
            system_tags=cls._project_system_tags,
        )

    @classmethod
    def create(
        cls,
        project_name: str,
        task_name: str,
        repo: str = None,
        branch: Optional[str] = None,
        commit: Optional[str] = None,
        script: Optional[str] = None,
        working_directory: Optional[str] = None,
        packages: Optional[Union[bool, Sequence[str]]] = None,
        requirements_file: Optional[Union[str, Path]] = None,
        docker: Optional[str] = None,
        docker_args: Optional[str] = None,
        docker_bash_setup_script: Optional[str] = None,
        argparse_args: Optional[Sequence[Tuple[str, str]]] = None,
        force_single_script_file: bool = False,
        version: Optional[str] = None,
        add_run_number: bool = True,
        binary: Optional[str] = None,
        module: Optional[str] = None,
        detect_repository: bool = True
    ) -> "PipelineController":
        """
        Manually create and populate a new Pipeline in the system.
        Supports pipelines from functions, decorators and tasks.

        :param project_name: Set the project name for the pipeline.
        :param task_name: Set the name of the remote pipeline..
        :param repo: Remote URL for the repository to use, or path to local copy of the git repository.
            Example: 'https://github.com/allegroai/clearml.git' or '~/project/repo'. If ``repo`` is specified, then
            the ``script`` parameter must also be specified
        :param branch: Select specific repository branch/tag (implies the latest commit from the branch)
        :param commit: Select specific commit ID to use (default: latest commit,
            or when used with local repository matching the local commit ID)
        :param script: Specify the entry point script for the remote execution. When used in tandem with
            remote git repository the script should be a relative path inside the repository,
            for example: './source/train.py' . When used with local repository path it supports a
            direct path to a file inside the local repository itself, for example: '~/project/source/train.py'
        :param working_directory: Working directory to launch the script from. Default: repository root folder.
            Relative to repo root or local folder.
        :param packages: Manually specify a list of required packages. Example: ``["tqdm>=2.1", "scikit-learn"]``
            or `True` to automatically create requirements
            based on locally installed packages (repository must be local).
            Pass an empty string to not install any packages (not even from the repository)
        :param requirements_file: Specify requirements.txt file to install when setting the session.
            If not provided, the requirements.txt from the repository will be used.
        :param docker: Select the docker image to be executed in by the remote session
        :param docker_args: Add docker arguments, pass a single string
        :param docker_bash_setup_script: Add bash script to be executed
            inside the docker before setting up the Task's environment
        :param argparse_args: Arguments to pass to the remote execution, list of string pairs (argument, value)
            Notice, only supported if the codebase itself uses argparse.ArgumentParser
        :param force_single_script_file: If True, do not auto-detect local repository
        :param binary: Binary used to launch the pipeline
        :param module: If specified instead of executing `script`, a module named `module` is executed.
            Implies script is empty. Module can contain multiple argument for execution,
            for example: module="my.module arg1 arg2"
        :param detect_repository: If True, detect the repository if no repository has been specified.
            If False, don't detect repository under any circumstance. Ignored if `repo` is specified

        :return: The newly created PipelineController
        """
        pipeline_project_args = cls._create_pipeline_project_args(name=task_name, project=project_name)
        pipeline_controller = Task.create(
            project_name=pipeline_project_args["project_name"],
            task_name=pipeline_project_args["task_name"],
            task_type=Task.TaskTypes.controller,
            repo=repo,
            branch=branch,
            commit=commit,
            script=script,
            working_directory=working_directory,
            packages=packages,
            requirements_file=requirements_file,
            docker=docker,
            docker_args=docker_args,
            docker_bash_setup_script=docker_bash_setup_script,
            argparse_args=argparse_args,
            add_task_init_call=False,
            force_single_script_file=force_single_script_file,
            binary=binary,
            module=module,
            detect_repository=detect_repository
        )
        cls._create_pipeline_projects(
            task=pipeline_controller,
            parent_project=pipeline_project_args["parent_project"],
            project_name=pipeline_project_args["project_name"],
        )
        pipeline_controller.set_system_tags((pipeline_controller.get_system_tags() or []) + [cls._tag])
        pipeline_controller.set_user_properties(version=version or cls._default_pipeline_version)
        if add_run_number:
            cls._add_pipeline_name_run_number(pipeline_controller)
        return cls._create_pipeline_controller_from_task(pipeline_controller)

    @classmethod
    def clone(
        cls,
        pipeline_controller: Union["PipelineController", str],
        name: Optional[str] = None,
        comment: Optional[str] = None,
        parent: Optional[str] = None,
        project: Optional[str] = None,
        version: Optional[str] = None,
    ) -> "PipelineController":
        """
        Create a duplicate (a clone) of a pipeline (experiment). The status of the cloned pipeline is ``Draft``
        and modifiable.

        :param str pipeline_controller: The pipeline to clone. Specify a PipelineController object or an ID.
        :param str name: The name of the new cloned pipeline.
        :param str comment: A comment / description for the new cloned pipeline.
        :param str parent: The ID of the parent Task of the new pipeline.

          - If ``parent`` is not specified, then ``parent`` is set to ``source_task.parent``.
          - If ``parent`` is not specified and ``source_task.parent`` is not available,
          then ``parent`` set to ``source_task``.

        :param str project: The project name in which to create the new pipeline.
            If ``None``, the clone inherits the original pipeline's project
        :param str version: The version of the new cloned pipeline. If ``None``, the clone
            inherits the original pipeline's version

        :return: The new cloned PipelineController
        """
        if isinstance(pipeline_controller, six.string_types):
            pipeline_controller = Task.get_task(task_id=pipeline_controller)
        elif isinstance(pipeline_controller, PipelineController):
            pipeline_controller = pipeline_controller.task

        if project or name:
            pipeline_project_args = cls._create_pipeline_project_args(
                name=name or pipeline_controller.name,
                project=project or pipeline_controller.get_project_name(),
            )
            project = cls._create_pipeline_projects(
                task=pipeline_controller,
                parent_project=pipeline_project_args["parent_project"],
                project_name=pipeline_project_args["project_name"],
            )
            name = pipeline_project_args["task_name"]
        cloned_controller = Task.clone(
            source_task=pipeline_controller,
            name=name,
            comment=comment,
            parent=parent,
            project=project,
        )
        if version:
            cloned_controller.set_user_properties(version=version)
        return cls._create_pipeline_controller_from_task(cloned_controller)

    @classmethod
    def enqueue(
        cls,
        pipeline_controller: Union["PipelineController", str],
        queue_name: Optional[str] = None,
        queue_id: Optional[str] = None,
        force: bool = False,
    ) -> Any:
        """
        Enqueue a PipelineController for execution, by adding it to an execution queue.

        .. note::
           A worker daemon must be listening at the queue for the worker to fetch the Task and execute it,
           see "ClearML Agent" in the ClearML Documentation.

        :param pipeline_controller: The PipelineController to enqueue. Specify a PipelineController object or PipelineController ID
        :param queue_name: The name of the queue. If not specified, then ``queue_id`` must be specified.
        :param queue_id: The ID of the queue. If not specified, then ``queue_name`` must be specified.
        :param bool force: If True, reset the PipelineController if necessary before enqueuing it

        :return: An enqueue JSON response.

            .. code-block:: javascript

               {
                    "queued": 1,
                    "updated": 1,
                    "fields": {
                        "status": "queued",
                        "status_reason": "",
                        "status_message": "",
                        "status_changed": "2020-02-24T15:05:35.426770+00:00",
                        "last_update": "2020-02-24T15:05:35.426770+00:00",
                        "execution.queue": "2bd96ab2d9e54b578cc2fb195e52c7cf"
                        }
                }

            - ``queued``  - The number of Tasks enqueued (an integer or ``null``).
            - ``updated`` - The number of Tasks updated (an integer or ``null``).
            - ``fields``

              - ``status`` - The status of the experiment.
              - ``status_reason`` - The reason for the last status change.
              - ``status_message`` - Information about the status.
              - ``status_changed`` - The last status change date and time (ISO 8601 format).
              - ``last_update`` - The last Task update time, including Task creation, update, change, or events for this task (ISO 8601 format).
              - ``execution.queue`` - The ID of the queue where the Task is enqueued. ``null`` indicates not enqueued.
        """
        pipeline_controller = (
            pipeline_controller
            if isinstance(pipeline_controller, PipelineController)
            else cls.get(pipeline_id=pipeline_controller)
        )
        return Task.enqueue(
            pipeline_controller._task,
            queue_name=queue_name,
            queue_id=queue_id,
            force=force,
        )

    @classmethod
    def get(
        cls,
        pipeline_id: Optional[str] = None,
        pipeline_project: Optional[str] = None,
        pipeline_name: Optional[str] = None,
        pipeline_version: Optional[str] = None,
        pipeline_tags: Optional[Sequence[str]] = None,
        shallow_search: bool = False,
    ) -> "PipelineController":
        """
        Get a specific PipelineController. If multiple pipeline controllers are found, the pipeline controller
        with the highest semantic version is returned. If no semantic version is found, the most recently
        updated pipeline controller is returned. This function raises aan Exception if no pipeline controller
        was found

        Note: In order to run the pipeline controller returned by this function, use PipelineController.enqueue

        :param pipeline_id: Requested PipelineController ID
        :param pipeline_project: Requested PipelineController project
        :param pipeline_name: Requested PipelineController name
        :param pipeline_tags: Requested PipelineController tags (list of tag strings)
        :param shallow_search: If True, search only the first 500 results (first page)
        """
        mutually_exclusive(
            pipeline_id=pipeline_id,
            pipeline_project=pipeline_project,
            _require_at_least_one=False,
        )
        mutually_exclusive(
            pipeline_id=pipeline_id,
            pipeline_name=pipeline_name,
            _require_at_least_one=False,
        )
        if not pipeline_id:
            pipeline_project_hidden = "{}/{}/{}".format(pipeline_project, cls._project_section, pipeline_name)
            name_with_runtime_number_regex = r"^{}( #[0-9]+)*$".format(re.escape(pipeline_name))
            pipelines = Task._query_tasks(
                pipeline_project=[pipeline_project_hidden],
                task_name=name_with_runtime_number_regex,
                fetch_only_first_page=False if not pipeline_version else shallow_search,
                only_fields=["id"] if not pipeline_version else ["id", "runtime.version"],
                system_tags=[cls._tag],
                order_by=["-last_update"],
                tags=pipeline_tags,
                search_hidden=True,
                _allow_extra_fields_=True,
            )
            if pipelines:
                if not pipeline_version:
                    pipeline_id = pipelines[0].id
                    current_version = None
                    for pipeline in pipelines:
                        if not pipeline.runtime:
                            continue
                        candidate_version = pipeline.runtime.get("version")
                        if not candidate_version or not Version.is_valid_version_string(candidate_version):
                            continue
                        if not current_version or Version(candidate_version) > current_version:
                            current_version = Version(candidate_version)
                            pipeline_id = pipeline.id
                else:
                    for pipeline in pipelines:
                        if pipeline.runtime.get("version") == pipeline_version:
                            pipeline_id = pipeline.id
                            break
            if not pipeline_id:
                error_msg = "Could not find dataset with pipeline_project={}, pipeline_name={}".format(
                    pipeline_project, pipeline_name
                )
                if pipeline_version:
                    error_msg += ", pipeline_version={}".format(pipeline_version)
                raise ValueError(error_msg)
        pipeline_task = Task.get_task(task_id=pipeline_id)
        return cls._create_pipeline_controller_from_task(pipeline_task)

    @classmethod
    def _create_pipeline_controller_from_task(cls, pipeline_task: Task) -> "PipelineController":
        pipeline_object = cls.__new__(cls)
        pipeline_object._task = pipeline_task
        pipeline_object._nodes = {}
        pipeline_object._running_nodes = []
        pipeline_object._version = pipeline_task._get_runtime_properties().get("version")
        try:
            pipeline_object._deserialize(pipeline_task._get_configuration_dict(cls._config_section), force=True)
        except Exception:
            pass
        return pipeline_object

    @property
    def task(self) -> Task:
        return self._task

    @property
    def id(self) -> str:
        return self._task.id

    @property
    def tags(self) -> List[str]:
        return self._task.get_tags() or []

    @property
    def version(self) -> str:
        return self._version

    def add_tags(self, tags: Union[Sequence[str], str]) -> None:
        """
        Add tags to this pipeline. Old tags are not deleted.
        When executing a Pipeline remotely
        (i.e. launching the pipeline from the UI/enqueuing it), this method has no effect.

        :param tags: A list of tags for this pipeline.
        """
        if not self._task:
            return  # should not actually happen
        self._task.add_tags(tags)

    def _create_task_from_function(
        self,
        docker: Optional[str],
        docker_args: Optional[str],
        docker_bash_setup_script: Optional[str],
        function: Callable,
        function_input_artifacts: Dict[str, str],
        function_kwargs: Dict[str, Any],
        function_return: List[str],
        auto_connect_frameworks: Optional[dict],
        auto_connect_arg_parser: Optional[dict],
        packages: Optional[Union[bool, str, Sequence[str]]],
        project_name: Optional[str],
        task_name: Optional[str],
        task_type: Optional[str],
        repo: Optional[str],
        branch: Optional[str],
        commit: Optional[str],
        helper_functions: Optional[Sequence[Callable]],
        output_uri: Optional[Union[str, bool]] = None,
        working_dir: Optional[str] = None,
    ) -> dict:
        task_definition = CreateFromFunction.create_task_from_function(
            a_function=function,
            function_kwargs=function_kwargs or None,
            function_input_artifacts=function_input_artifacts,
            function_return=function_return,
            project_name=project_name,
            task_name=task_name,
            task_type=task_type,
            auto_connect_frameworks=auto_connect_frameworks,
            auto_connect_arg_parser=auto_connect_arg_parser,
            repo=repo,
            branch=branch,
            commit=commit,
            packages=packages,
            docker=docker,
            docker_args=docker_args,
            docker_bash_setup_script=docker_bash_setup_script,
            output_uri=output_uri,
            helper_functions=helper_functions,
            dry_run=True,
            task_template_header=self._task_template_header,
            artifact_serialization_function=self._artifact_serialization_function,
            artifact_deserialization_function=self._artifact_deserialization_function,
            skip_global_imports=self._skip_global_imports,
            working_dir=working_dir,
        )
        return task_definition

    def _start(
        self,
        step_task_created_callback: Optional[
            Callable[["PipelineController", "PipelineController.Node", dict], bool]
        ] = None,  # noqa
        step_task_completed_callback: Optional[
            Callable[["PipelineController", "PipelineController.Node"], None]
        ] = None,  # noqa
        wait: bool = True,
    ) -> bool:
        """
        Start the pipeline controller.
        If the calling process is stopped, then the controller stops as well.

        :param Callable step_task_created_callback: Callback function, called when a step (Task) is created
            and before it is sent for execution. Allows a user to modify the Task before launch.
            Use `node.job` to access the ClearmlJob object, or `node.job.task` to directly access the Task object.
            `parameters` are the configuration arguments passed to the ClearmlJob.

            If the callback returned value is `False`,
            the Node is skipped and so is any node in the DAG that relies on this node.

            Notice the `parameters` are already parsed,
            e.g. ``${step1.parameters.Args/param}`` is replaced with relevant value.

            .. code-block:: py

                def step_created_callback(
                    pipeline,             # type: PipelineController,
                    node,                 # type: PipelineController.Node,
                    parameters,           # type: dict
                ):
                    pass

        :param Callable step_task_completed_callback: Callback function, called when a step (Task) is completed
            and other jobs are executed. Allows a user to modify the Task status after completion.

            .. code-block:: py

                def step_completed_callback(
                    pipeline,             # type: PipelineController,
                    node,                 # type: PipelineController.Node,
                ):
                    pass
        :param wait: If True (default), start the pipeline controller, return only
        after the pipeline is done (completed/aborted/failed)

        :return: True, if the controller started. False, if the controller did not start.

        """
        if self._thread:
            return True

        self._prepare_pipeline(step_task_created_callback, step_task_completed_callback)
        self._thread = Thread(target=self._daemon)
        self._thread.daemon = True
        self._thread.start()

        if wait:
            self.wait()
            self.stop()

        return True

    def _prepare_pipeline(
        self,
        step_task_created_callback: Optional[
            Callable[["PipelineController", "PipelineController.Node", dict], bool]
        ] = None,  # noqa
        step_task_completed_callback: Optional[
            Callable[["PipelineController", "PipelineController.Node"], None]
        ] = None,  # noqa
    ) -> None:
        # type (...) -> None

        params, pipeline_dag = self._serialize_pipeline_task()
        # deserialize back pipeline state
        if not params["continue_pipeline"]:
            for k in pipeline_dag:
                pipeline_dag[k]["executed"] = None
                pipeline_dag[k]["job_started"] = None
                pipeline_dag[k]["job_ended"] = None
        self._default_execution_queue = params["default_queue"]
        self._add_pipeline_tags = params["add_pipeline_tags"]
        self._target_project = params["target_project"] or ""
        self._deserialize(pipeline_dag)
        # if we continue the pipeline, make sure that we re-execute failed tasks
        if params["continue_pipeline"]:
            for node in list(self._nodes.values()):
                if node.executed is False:
                    node.executed = None
        if not self._verify():
            raise ValueError(
                "Failed verifying pipeline execution graph, it has either inaccessible nodes, or contains cycles"
            )
        self.update_execution_plot()
        self._start_time = time()
        self._stop_event = Event()
        self._experiment_created_cb = step_task_created_callback
        self._experiment_completed_cb = step_task_completed_callback

    def _serialize_pipeline_task(self) -> (dict, dict):
        """
        Serialize current pipeline state into the main Task

        :return: params, pipeline_dag
        """
        params = {
            "default_queue": self._default_execution_queue,
            "add_pipeline_tags": self._add_pipeline_tags,
            "target_project": self._target_project,
        }
        pipeline_dag = self._serialize()

        # serialize pipeline state
        if self._task and self._auto_connect_task:
            # check if we are either running locally or that we are running remotely,
            # but we have no configuration, so we need to act as if this is a local run and create everything
            if self._task.running_locally() or self._task.get_configuration_object(name=self._config_section) is None:
                # noinspection PyProtectedMember
                self._task._set_configuration(
                    name=self._config_section,
                    config_type="dictionary",
                    config_text=json.dumps(pipeline_dag, indent=2),
                )
                args_map_inversed = {}
                for section, arg_list in self._args_map.items():
                    for arg in arg_list:
                        args_map_inversed[arg] = section
                pipeline_args = flatten_dictionary(self._pipeline_args)
                # noinspection PyProtectedMember
                self._task._set_parameters(
                    {
                        "{}/{}".format(args_map_inversed.get(k, self._args_section), k): v
                        for k, v in pipeline_args.items()
                    },
                    __parameters_descriptions=self._pipeline_args_desc,
                    __parameters_types=self._pipeline_args_type,
                    __update=True,
                )
                self._task.connect(params, name=self._pipeline_section)
                params["continue_pipeline"] = False

                # make sure we have a unique version number (auto bump version if needed)
                # only needed when manually (from code) creating pipelines
                self._handle_pipeline_version()

                # noinspection PyProtectedMember
                pipeline_hash = self._get_task_hash()

                # noinspection PyProtectedMember
                self._task._set_runtime_properties(
                    {
                        self._runtime_property_hash: "{}:{}".format(pipeline_hash, self._version),
                        "version": self._version,
                    }
                )
                self._task.set_user_properties(version=self._version)
            else:
                self._task.connect_configuration(pipeline_dag, name=self._config_section)
                connected_args = set()
                new_pipeline_args = {}
                for section, arg_list in self._args_map.items():
                    mutable_dict = {arg: self._pipeline_args.get(arg) for arg in arg_list}
                    self._task.connect(mutable_dict, name=section)
                    new_pipeline_args.update(mutable_dict)
                    connected_args.update(arg_list)
                mutable_dict = {k: v for k, v in self._pipeline_args.items() if k not in connected_args}
                self._task.connect(mutable_dict, name=self._args_section)
                new_pipeline_args.update(mutable_dict)
                self._pipeline_args = new_pipeline_args
                self._task.connect(params, name=self._pipeline_section)
                # noinspection PyProtectedMember
                if self._task._get_runtime_properties().get(self._runtime_property_hash):
                    params["continue_pipeline"] = True
                else:
                    # noinspection PyProtectedMember
                    pipeline_hash = ClearmlJob._create_task_hash(self._task)
                    # noinspection PyProtectedMember
                    self._task._set_runtime_properties(
                        {
                            self._runtime_property_hash: "{}:{}".format(pipeline_hash, self._version),
                        }
                    )
                    params["continue_pipeline"] = False

        return params, pipeline_dag

    def _handle_pipeline_version(self) -> None:
        if not self._version:
            # noinspection PyProtectedMember
            self._version = self._task._get_runtime_properties().get("version")
            if not self._version:
                previous_pipeline_tasks = Task._query_tasks(
                    project=[self._task.project],
                    fetch_only_first_page=True,
                    only_fields=["runtime.version"],
                    order_by=["-last_update"],
                    system_tags=[self._tag],
                    search_hidden=True,
                    _allow_extra_fields_=True,
                )
                for previous_pipeline_task in previous_pipeline_tasks:
                    if previous_pipeline_task.runtime.get("version"):
                        self._version = str(Version(previous_pipeline_task.runtime.get("version")).get_next_version())
                        break
        self._version = self._version or self._default_pipeline_version

    def _get_task_hash(self) -> str:
        params_override = dict(**(self._task.get_parameters() or {}))
        params_override.pop("properties/version", None)
        # dag state without status / states
        nodes_items = list(self._nodes.items())
        dag = {
            name: {
                k: v
                for k, v in node.__dict__.items()
                if k
                not in (
                    "job",
                    "name",
                    "task_factory_func",
                    "executed",
                    "status",
                    "job_started",
                    "job_ended",
                    "skip_job",
                )
            }
            for name, node in nodes_items
        }

        # get all configurations (as dict of strings for hashing)
        configurations_override = dict(**self._task.get_configuration_objects())
        # store as text so we can hash it later
        configurations_override[self._config_section] = json.dumps(dag)

        # noinspection PyProtectedMember
        pipeline_hash = ClearmlJob._create_task_hash(
            self._task,
            params_override=params_override,
            configurations_override=configurations_override,
        )
        return pipeline_hash

    def _serialize(self) -> dict:
        """
        Store the definition of the pipeline DAG into a dictionary.
        This dictionary will be used to store the DAG as a configuration on the Task
        :return:
        """
        nodes_items = list(self._nodes.items())
        dag = {
            name: dict((k, v) for k, v in node.__dict__.items() if k not in ("job", "name", "task_factory_func"))
            for name, node in nodes_items
        }
        # update state for presentation only
        for name, node in nodes_items:
            dag[name]["job_id"] = node.executed or (node.job.task_id() if node.job else None)

        return dag

    def _deserialize(self, dag_dict: dict, force: bool = False) -> ():
        """
        Restore the DAG from a dictionary.
        This will be used to create the DAG from the dict stored on the Task, when running remotely.
        :return:
        """
        # if we always want to load the pipeline DAG from code, we are skipping the deserialization step
        if not force and self._always_create_from_code:
            return

        # if we do not clone the Task, only merge the parts we can override.
        for name in list(self._nodes.keys()):
            if not self._nodes[name].clone_task and name in dag_dict and not dag_dict[name].get("clone_task"):
                for k in (
                    "queue",
                    "parents",
                    "timeout",
                    "parameters",
                    "configurations",
                    "task_overrides",
                    "executed",
                    "job_started",
                    "job_ended",
                ):
                    setattr(
                        self._nodes[name],
                        k,
                        dag_dict[name].get(k) or type(getattr(self._nodes[name], k))(),
                    )

        # if we do clone the Task deserialize everything, except the function creating
        self._nodes = {
            k: self.Node(name=k, **{kk: vv for kk, vv in v.items() if kk not in ("job_id",)})
            if k not in self._nodes or (v.get("base_task_id") and v.get("clone_task"))
            else self._nodes[k]
            for k, v in dag_dict.items()
        }

        # set the task_factory_func for each cloned node
        for node in list(self._nodes.values()):
            if not node.base_task_id and not node.task_factory_func and node.job_code_section:
                if node.job_code_section in self._nodes:
                    func = self._nodes[node.job_code_section].task_factory_func
                    if func:
                        node.task_factory_func = func

    def _has_stored_configuration(self) -> bool:
        """
        Return True if we are running remotely, and we have stored configuration on the Task
        """
        if self._auto_connect_task and self._task and not self._task.running_locally() and self._task.is_main_task():
            stored_config = self._task.get_configuration_object(self._config_section)
            return bool(stored_config)

        return False

    def _verify(self) -> bool:
        """
        Verify the DAG, (i.e. no cycles and no missing parents)
        On error raise ValueError with verification details

        :return: return True iff DAG has no errors
        """
        # verify nodes
        for node in list(self._nodes.values()):
            # raise value error if not verified
            self._verify_node(node)

        # check the dag itself
        if not self._verify_dag():
            return False

        return True

    def _verify_node(self, node: "PipelineController.Node") -> bool:
        """
        Raise ValueError on verification errors

        :return: Return True iff the specific node is verified
        """
        if not node.base_task_id and not node.task_factory_func:
            raise ValueError("Node '{}', base_task_id is empty".format(node.name))

        if not self._default_execution_queue and not node.queue:
            raise ValueError(
                "Node '{}' missing execution queue, "
                "no default queue defined and no specific node queue defined".format(node.name)
            )

        task = node.task_factory_func or Task.get_task(task_id=node.base_task_id)
        if not task:
            raise ValueError("Node '{}', base_task_id={} is invalid".format(node.name, node.base_task_id))

        pattern = self._step_ref_pattern

        # verify original node parents
        if node.parents and not all(isinstance(p, str) and p in self._nodes for p in node.parents):
            raise ValueError("Node '{}', parents={} is invalid".format(node.name, node.parents))

        parents = set()
        for k, v in node.parameters.items():
            if isinstance(v, str):
                for g in pattern.findall(v):
                    ref_step = self.__verify_step_reference(node, g)
                    if ref_step:
                        parents.add(ref_step)
            # verify we have a section name
            if "/" not in k:
                raise ValueError(
                    'Section name is missing in parameter "{}", '
                    "parameters should be in the form of "
                    '"`section-name`/parameter", example: "Args/param"'.format(v)
                )

        if parents and parents != set(node.parents or []):
            parents = parents - set(node.parents or [])
            getLogger("clearml.automation.controller").info(
                'Node "{}" missing parent reference, adding: {}'.format(node.name, parents)
            )
            node.parents = (node.parents or []) + list(parents)

        # verify and fix monitoring sections:
        def _verify_monitors(
            monitors: Union[List[Union[str, Tuple[Any, Any]]], None],
            monitor_type: str,
            nested_pairs: bool = False,
        ) -> List[Tuple[Union[str, Tuple[str, str]], Union[str, Tuple[str, str]]]]:
            if not monitors:
                return monitors

            if nested_pairs:
                if not all(isinstance(x, (list, tuple)) and x for x in monitors):
                    raise ValueError("{} should be a list of tuples, found: {}".format(monitor_type, monitors))
                # convert single pair into a pair of pairs:
                conformed_monitors = [pair if isinstance(pair[0], (list, tuple)) else (pair, pair) for pair in monitors]
                # verify the pair of pairs
                if not all(
                    isinstance(x[0][0], str)
                    and isinstance(x[0][1], str)
                    and isinstance(x[1][0], str)
                    and isinstance(x[1][1], str)
                    for x in conformed_monitors
                ):
                    raise ValueError("{} should be a list of tuples, found: {}".format(monitor_type, monitors))
            else:
                # verify a list of tuples
                if not all(isinstance(x, (list, tuple, str)) and x for x in monitors):
                    raise ValueError("{} should be a list of tuples, found: {}".format(monitor_type, monitors))
                # convert single str into a pair of pairs:
                conformed_monitors = [pair if isinstance(pair, (list, tuple)) else (pair, pair) for pair in monitors]
                # verify the pair of pairs
                if not all(isinstance(x[0], str) and isinstance(x[1], str) for x in conformed_monitors):
                    raise ValueError("{} should be a list of tuples, found: {}".format(monitor_type, monitors))

            return conformed_monitors

        # verify and fix monitoring sections:
        node.monitor_metrics = _verify_monitors(node.monitor_metrics, "monitor_metrics", nested_pairs=True)
        node.monitor_artifacts = _verify_monitors(node.monitor_artifacts, "monitor_artifacts")
        node.monitor_models = _verify_monitors(node.monitor_models, "monitor_models")

        return True

    def _verify_dag(self) -> bool:
        """
        :return: True iff the pipeline dag is fully accessible and contains no cycles
        """
        visited = set()
        prev_visited = None
        while prev_visited != visited:
            prev_visited = copy(visited)
            for k, node in list(self._nodes.items()):
                if k in visited:
                    continue
                if any(p == node.name for p in node.parents or []):
                    # node cannot have itself as parent
                    return False
                if not all(p in visited for p in node.parents or []):
                    continue
                visited.add(k)
        # return False if we did not cover all the nodes
        return not bool(set(self._nodes.keys()) - visited)

    def _add_function_step(
        self,
        name: str,
        function: Callable,
        function_kwargs: Optional[Dict[str, Any]] = None,
        function_return: Optional[List[str]] = None,
        project_name: Optional[str] = None,
        task_name: Optional[str] = None,
        task_type: Optional[str] = None,
        auto_connect_frameworks: Optional[dict] = None,
        auto_connect_arg_parser: Optional[dict] = None,
        packages: Optional[Union[bool, str, Sequence[str]]] = None,
        repo: Optional[str] = None,
        repo_branch: Optional[str] = None,
        repo_commit: Optional[str] = None,
        helper_functions: Optional[Sequence[Callable]] = None,
        docker: Optional[str] = None,
        docker_args: Optional[str] = None,
        docker_bash_setup_script: Optional[str] = None,
        parents: Optional[Sequence[str]] = None,
        execution_queue: Optional[str] = None,
        monitor_metrics: Optional[List[Union[Tuple[str, str], Tuple]]] = None,
        monitor_artifacts: Optional[List[Union[str, Tuple[str, str]]]] = None,
        monitor_models: Optional[List[Union[str, Tuple[str, str]]]] = None,
        time_limit: Optional[float] = None,
        continue_on_fail: bool = False,
        pre_execute_callback: Optional[
            Callable[["PipelineController", "PipelineController.Node", dict], bool]
        ] = None,  # noqa
        post_execute_callback: Optional[Callable[["PipelineController", "PipelineController.Node"], None]] = None,
        # noqa
        cache_executed_step: bool = False,
        retry_on_failure: Optional[
            Union[
                int,
                Callable[["PipelineController", "PipelineController.Node", int], bool],
            ]
        ] = None,  # noqa
        status_change_callback: Optional[
            Callable[["PipelineController", "PipelineController.Node", str], None]
        ] = None,  # noqa
        tags: Optional[Union[str, Sequence[str]]] = None,
        output_uri: Optional[Union[str, bool]] = None,
        draft: Optional[bool] = False,
        working_dir: Optional[str] = None,
        continue_behaviour: Optional[dict] = None,
        stage: Optional[str] = None
    ) -> bool:
        """
        Create a Task from a function, including wrapping the function input arguments
        into the hyperparameter section as kwargs, and storing function results as named artifacts

        Example:

        .. code-block:: py

            def mock_func(a=6, b=9):
                c = a*b
                print(a, b, c)
                return c, c**2

            create_task_from_function(mock_func, function_return=['mul', 'square'])

        Example arguments from other Tasks (artifact):

        .. code-block:: py

            def mock_func(matrix_np):
                c = matrix_np*matrix_np
                print(matrix_np, c)
                return c

            create_task_from_function(
                mock_func,
                function_kwargs={'matrix_np': 'aabb1122.previous_matrix'},
                function_return=['square_matrix']
            )

        :param name: Unique of the step. For example `stage1`
        :param function: A global function to convert into a standalone Task
        :param function_kwargs: Optional, provide subset of function arguments and default values to expose.
            If not provided automatically take all function arguments & defaults
            Optional, pass input arguments to the function from other Tasks's output artifact.
            Example argument named `numpy_matrix` from Task ID `aabbcc` artifact name `answer`:
            ``{'numpy_matrix': 'aabbcc.answer'}``
        :param function_return: Provide a list of names for all the results.
            If not provided, no results will be stored as artifacts.
        :param project_name: Set the project name for the task. Required if base_task_id is None.
        :param task_name: Set the name of the remote task, if not provided use `name` argument.
        :param task_type: Optional, The task type to be created. Supported values: 'training', 'testing', 'inference',
            'data_processing', 'application', 'monitor', 'controller', 'optimizer', 'service', 'qc', 'custom'
        :param auto_connect_frameworks: Control the frameworks auto connect, see `Task.init` auto_connect_frameworks
        :param auto_connect_arg_parser: Control the ArgParser auto connect, see `Task.init` auto_connect_arg_parser
        :param packages: Manually specify a list of required packages or a local requirements.txt file.
            Example: ["tqdm>=2.1", "scikit-learn"] or "./requirements.txt"
            If not provided, packages are automatically added based on the imports used in the function.
            Use `False` to install requirements from "requirements.txt" inside your git repository
        :param repo: Optional, specify a repository to attach to the function, when remotely executing.
            Allow users to execute the function inside the specified repository, enabling to load modules/script
            from a repository Notice the execution work directory will be the repository root folder.
            Supports both git repo url link, and local repository path.
            Example remote url: 'https://github.com/user/repo.git'
            Example local repo copy: './repo' -> will automatically store the remote
            repo url and commit ID based on the locally cloned copy
        :param repo_branch: Optional, specify the remote repository branch (Ignored, if local repo path is used)
        :param repo_commit: Optional, specify the repository commit ID (Ignored, if local repo path is used)
        :param helper_functions: Optional, a list of helper functions to make available
            for the standalone function Task.
        :param docker: Select the docker image to be executed in by the remote session
        :param docker_args: Add docker arguments, pass a single string
        :param docker_bash_setup_script: Add bash script to be executed
            inside the docker before setting up the Task's environment
        :param parents: Optional list of parent nodes in the DAG.
            The current step in the pipeline will be sent for execution only after all the parent nodes
            have been executed successfully.
        :param execution_queue: Optional, the queue to use for executing this specific step.
            If not provided, the task will be sent to the default execution queue, as defined on the class
        :param monitor_metrics: Optional, log the step's metrics on the pipeline Task.
            Format is a list of pairs metric (title, series) to log:
                [(step_metric_title, step_metric_series), ]
                Example: [('test', 'accuracy'), ]
            Or a list of tuple pairs, to specify a different target metric for to use on the pipeline Task:
                [((step_metric_title, step_metric_series), (target_metric_title, target_metric_series)), ]
                Example: [[('test', 'accuracy'), ('model', 'accuracy')], ]
        :param monitor_artifacts: Optional, log the step's artifacts on the pipeline Task.
            Provided a list of artifact names existing on the step's Task, they will also appear on the Pipeline itself.
            Example: [('processed_data', 'final_processed_data'), ]
            Alternatively user can also provide a list of artifacts to monitor
            (target artifact name will be the same as original artifact name)
            Example: ['processed_data', ]
        :param monitor_models: Optional, log the step's output models on the pipeline Task.
            Provided a list of model names existing on the step's Task, they will also appear on the Pipeline itself.
            Example: [('model_weights', 'final_model_weights'), ]
            Alternatively user can also provide a list of models to monitor
            (target models name will be the same as original model)
            Example: ['model_weights', ]
            To select the latest (lexicographic) model use "model_*", or the last created model with just "*"
            Example:  ['model_weights_*', ]
        :param time_limit: Default None, no time limit.
            Step execution time limit, if exceeded the Task is aborted and the pipeline is stopped and marked failed.
        :param continue_on_fail: (Deprecated, use `continue_behaviour` instead).
            If True, failed step will not cause the pipeline to stop
            (or marked as failed). Notice, that steps that are connected (or indirectly connected)
            to the failed step will be skipped. Defaults to False
        :param pre_execute_callback: Callback function, called when the step (Task) is created,
            and before it is sent for execution. Allows a user to modify the Task before launch.
            Use `node.job` to access the ClearmlJob object, or `node.job.task` to directly access the Task object.
            ``parameters`` are the configuration arguments passed to the ClearmlJob.

            If the callback returned value is `False`,
            the Node is skipped and so is any node in the DAG that relies on this node.

            Notice the `parameters` are already parsed,
            e.g. ``${step1.parameters.Args/param}`` is replaced with relevant value.

            .. code-block:: py

                def step_created_callback(
                    pipeline,             # type: PipelineController,
                    node,                 # type: PipelineController.Node,
                    parameters,           # type: dict
                ):
                    pass

        :param post_execute_callback: Callback function, called when a step (Task) is completed
            and other jobs are executed. Allows a user to modify the Task status after completion.

            .. code-block:: py

                def step_completed_callback(
                    pipeline,             # type: PipelineController,
                    node,                 # type: PipelineController.Node,
                ):
                    pass

        :param cache_executed_step: If True, before launching the new step,
            after updating with the latest configuration, check if an exact Task with the same parameter/code
            was already executed. If it was found, use it instead of launching a new Task.
            Default: False, a new cloned copy of base_task is always used.
            Notice: If the git repo reference does not have a specific commit ID, the Task will never be used.

        :param retry_on_failure: Integer (number of retries) or Callback function that returns True to allow a retry
            - Integer: In case of node failure, retry the node the number of times indicated by this parameter.
            - Callable: A function called on node failure. Takes as parameters:
              the PipelineController instance, the PipelineController.Node that failed and an int
              representing the number of previous retries for the node that failed
              The function must return a `bool`: True if the node should be retried and False otherwise.
              If True, the node will be re-queued and the number of retries left will be decremented by 1.
              By default, if this callback is not specified, the function will be retried the number of
              times indicated by `retry_on_failure`.

              .. code-block:: py

                  def example_retry_on_failure_callback(pipeline, node, retries):
                      print(node.name, ' failed')
                      # allow up to 5 retries (total of 6 runs)
                      return retries < 5

        :param status_change_callback: Callback function, called when the status of a step (Task) changes.
            Use `node.job` to access the ClearmlJob object, or `node.job.task` to directly access the Task object.
            The signature of the function must look the following way:

            .. code-block:: py

                def status_change_callback(
                    pipeline,             # type: PipelineController,
                    node,                 # type: PipelineController.Node,
                    previous_status       # type: str
                ):
                    pass

        :param tags: A list of tags for the specific pipeline step.
            When executing a Pipeline remotely
            (i.e. launching the pipeline from the UI/enqueuing it), this method has no effect.
        :param output_uri: The storage / output url for this step. This is the default location for output
            models and other artifacts. Check Task.init reference docs for more info (output_uri is a parameter).
        :param draft: (default False). If True, the Task will be created as a draft task.
        :param working_dir:  Working directory to launch the step from.
        :param continue_behaviour: Controls whether the pipeline will continue running after a step failed/was aborted.
            Different behaviours can be set using a dictionary of boolean options. Supported options are:
              - continue_on_fail - If True, the pipeline will continue even if the step failed.
                 If False, the pipeline will stop
              - continue_on_abort - If True, the pipeline will continue even if the step was aborted.
                 If False, the pipeline will stop
              - skip_children_on_fail - If True, the children of this step will be skipped if it failed.
                 If False, the children will run even if this step failed.
                 Any parameters passed from the failed step to its children will default to None
              - skip_children_on_abort - If True, the children of this step will be skipped if it was aborted.
                 If False, the children will run even if this step was aborted.
                 Any parameters passed from the failed step to its children will default to None
              If the keys are not present in the dictionary, their values will default to True
        :param stage: Name of the stage. This parameter enables pipeline step grouping into stages

        :return: True if successful
        """
        # always store callback functions (even when running remotely)
        if pre_execute_callback:
            self._pre_step_callbacks[name] = pre_execute_callback
        if post_execute_callback:
            self._post_step_callbacks[name] = post_execute_callback
        if status_change_callback:
            self._status_change_callbacks[name] = status_change_callback

        self._verify_node_name(name)

        if output_uri is None:
            output_uri = self._output_uri

        function_input_artifacts = {}
        # go over function_kwargs, split it into string and input artifacts
        for k, v in function_kwargs.items():
            if v is None:
                continue
            if self._step_ref_pattern.match(str(v)):
                # check for step artifacts
                step, _, artifact = v[2:-1].partition(".")
                if step in self._nodes and artifact in self._nodes[step].return_artifacts:
                    function_input_artifacts[k] = "${{{}.id}}.{}".format(step, artifact)
                    continue
                # verify the reference only if we are running locally (on remote when we have multiple
                # steps from tasks the _nodes is till empty, only after deserializing we will have the full DAG)
                if self._task.running_locally():
                    self.__verify_step_reference(node=self.Node(name=name), step_ref_string=v)
            elif not isinstance(v, (float, int, bool, six.string_types)):
                function_input_artifacts[k] = "{}.{}.{}".format(self._task.id, name, k)
                self._upload_pipeline_artifact(artifact_name="{}.{}".format(name, k), artifact_object=v)

        function_kwargs = {k: v for k, v in function_kwargs.items() if k not in function_input_artifacts}
        parameters = {"{}/{}".format(CreateFromFunction.kwargs_section, k): v for k, v in function_kwargs.items()}
        if function_input_artifacts:
            parameters.update(
                {
                    "{}/{}".format(CreateFromFunction.input_artifact_section, k): str(v)
                    for k, v in function_input_artifacts.items()
                }
            )

        job_code_section = name
        task_name = task_name or name or None

        if self._mock_execution:
            project_name = project_name or self._get_target_project() or self._task.get_project_name()

            task_definition = self._create_task_from_function(
                docker,
                docker_args,
                docker_bash_setup_script,
                function,
                function_input_artifacts,
                function_kwargs,
                function_return,
                auto_connect_frameworks,
                auto_connect_arg_parser,
                packages,
                project_name,
                task_name,
                task_type,
                repo,
                repo_branch,
                repo_commit,
                helper_functions,
                output_uri=output_uri,
                working_dir=working_dir,
            )

        elif self._task.running_locally() or self._task.get_configuration_object(name=name) is None:
            project_name = project_name or self._get_target_project() or self._task.get_project_name()

            task_definition = self._create_task_from_function(
                docker,
                docker_args,
                docker_bash_setup_script,
                function,
                function_input_artifacts,
                function_kwargs,
                function_return,
                auto_connect_frameworks,
                auto_connect_arg_parser,
                packages,
                project_name,
                task_name,
                task_type,
                repo,
                repo_branch,
                repo_commit,
                helper_functions,
                output_uri=output_uri,
                working_dir=working_dir,
            )
            # update configuration with the task definitions
            # noinspection PyProtectedMember
            self._task._set_configuration(
                name=name,
                config_type="json",
                config_text=json.dumps(task_definition, indent=1),
            )
        else:
            # load task definition from configuration
            # noinspection PyProtectedMember
            config_text = self._task._get_configuration_text(name=name)
            task_definition = json.loads(config_text) if config_text else dict()

        def _create_task(_: Any) -> Task:
            a_task = Task.create(
                project_name=project_name,
                task_name=task_definition.get("name"),
                task_type=task_definition.get("type"),
            )
            # replace reference
            a_task.update_task(task_definition)

            if tags:
                a_task.add_tags(tags)

            if output_uri is not None:
                a_task.output_uri = output_uri

            return a_task

        self._nodes[name] = self.Node(
            name=name,
            base_task_id=None,
            parents=parents or [],
            queue=execution_queue,
            timeout=time_limit,
            parameters=parameters,
            clone_task=False,
            cache_executed_step=cache_executed_step,
            task_factory_func=_create_task,
            continue_on_fail=continue_on_fail,
            return_artifacts=function_return,
            monitor_artifacts=monitor_artifacts,
            monitor_metrics=monitor_metrics,
            monitor_models=monitor_models,
            job_code_section=job_code_section,
            explicit_docker_image=docker,
            output_uri=output_uri,
            draft=draft,
            continue_behaviour=continue_behaviour,
            stage=stage
        )
        self._retries[name] = 0
        self._retries_callbacks[name] = (
            retry_on_failure
            if callable(retry_on_failure)
            else (
                functools.partial(
                    self._default_retry_on_failure_callback,
                    max_retries=retry_on_failure,
                )
                if isinstance(retry_on_failure, int)
                else self._retry_on_failure_callback
            )
        )

        return True

    def _relaunch_node(self, node: "PipelineController.Node") -> None:
        if not node.job:
            getLogger("clearml.automation.controller").warning(
                "Could not relaunch node {} (job object is missing)".format(node.name)
            )
            return
        self._retries[node.name] = self._retries.get(node.name, 0) + 1
        getLogger("clearml.automation.controller").warning(
            "Node '{}' failed. Retrying... (this is retry number {})".format(node.name, self._retries[node.name])
        )
        node.job.task.mark_stopped(force=True, status_message=self._relaunch_status_message)
        node.job.task.set_progress(0)
        node.job.task.get_logger().report_text(
            "\nNode '{}' failed. Retrying... (this is retry number {})\n".format(node.name, self._retries[node.name])
        )
        parsed_queue_name = self._parse_step_ref(node.queue)
        node.job.launch(queue_name=parsed_queue_name or self._default_execution_queue)

    def _launch_node(self, node: "PipelineController.Node") -> ():
        """
        Launch a single node (create and enqueue a ClearmlJob)

        :param node: Node to launch
        :return: Return True if a new job was launched
        """
        # clear state if we are creating a new job
        if not node.job:
            node.job_started = None
            node.job_ended = None
            node.job_type = None

        if node.job or node.executed:
            print("Skipping cached/executed step [{}]".format(node.name))
            return False

        print("Launching step [{}]".format(node.name))

        updated_hyper_parameters = {}
        for k, v in node.parameters.items():
            updated_hyper_parameters[k] = self._parse_step_ref(v, recursive=node.recursively_parse_parameters)

        task_overrides = self._parse_task_overrides(node.task_overrides) if node.task_overrides else None

        extra_args = dict()
        extra_args["project"] = self._get_target_project(return_project_id=True) or None
        # set Task name to match job name
        if self._pipeline_as_sub_project():
            extra_args["name"] = node.name
        if node.explicit_docker_image:
            extra_args["explicit_docker_image"] = node.explicit_docker_image

        skip_node = None
        if self._pre_step_callbacks.get(node.name):
            skip_node = self._pre_step_callbacks[node.name](self, node, updated_hyper_parameters)

        if skip_node is False:
            node.skip_job = True
            return True

        task_id = node.base_task_id
        disable_clone_task = not node.clone_task
        task_factory_func_task = None
        if node.task_factory_func:
            # create Task
            task_factory_func_task = node.task_factory_func(node)
            task_id = task_factory_func_task.id
            disable_clone_task = True

        try:
            node.job = self._clearml_job_class(
                base_task_id=task_id,
                parameter_override=updated_hyper_parameters,
                configuration_overrides=node.configurations,
                tags=["{} {}".format(self._node_tag_prefix, self._task.id)]
                if self._add_pipeline_tags and self._task
                else None,
                parent=self._task.id if self._task else None,
                disable_clone_task=disable_clone_task,
                task_overrides=task_overrides,
                allow_caching=node.cache_executed_step,
                output_uri=node.output_uri,
                enable_local_imports=self._enable_local_imports,
                **extra_args,
            )
        except Exception:
            self._pipeline_task_status_failed = True
            raise

        node.job_started = None
        node.job_ended = None
        node.job_type = str(node.job.task.task_type)

        if self._experiment_created_cb:
            skip_node = self._experiment_created_cb(self, node, updated_hyper_parameters)

        if skip_node is False:
            # skipping node
            getLogger("clearml.automation.controller").warning("Skipping node {} on callback request".format(node))
            # delete the job we just created
            node.job.delete()
            node.skip_job = True
        elif node.job.is_cached_task():
            node.executed = node.job.task_id()
            if task_factory_func_task:
                task_factory_func_task.delete(raise_on_error=False)
            self._running_nodes.append(node.name)
        elif node.draft:
            self._running_nodes.append(node.name)
        else:
            self._running_nodes.append(node.name)

            parsed_queue_name = self._parse_step_ref(node.queue)
            return node.job.launch(queue_name=parsed_queue_name or self._default_execution_queue)

        return True

    def _update_execution_plot(self) -> ():
        """
        Update sankey diagram of the current pipeline
        Also update the controller Task artifact storing the DAG state (with all the nodes states)
        """
        if not self._task:
            return

        nodes = list(self._nodes.values())
        self._update_nodes_status()

        # update the configuration state, so that the UI is presents the correct state
        self._force_task_configuration_update()

        sankey_node = dict(
            label=[],
            color=[],
            hovertemplate="%{label}<extra></extra>",
            # customdata=[],
            # hovertemplate='%{label}<br />Hyper-Parameters:<br />%{customdata}<extra></extra>',
        )
        sankey_link = dict(
            source=[],
            target=[],
            value=[],
            # hovertemplate='%{target.label}<extra></extra>',
            hovertemplate="<extra></extra>",
        )
        visited = []
        node_params = []
        # update colors
        while nodes:
            next_nodes = []
            for node in nodes:
                if not all(p in visited for p in node.parents or []):
                    next_nodes.append(node)
                    continue
                visited.append(node.name)
                idx = len(visited) - 1
                parents = [visited.index(p) for p in node.parents or []]
                if node.job and node.job.task_parameter_override is not None:
                    node.job.task_parameter_override.update(node.parameters or {})
                node_params.append(
                    (
                        node.job.task_parameter_override
                        if node.job and node.job.task_parameter_override
                        else node.parameters
                    )
                    or {}
                )
                # sankey_node['label'].append(node.name)
                # sankey_node['customdata'].append(
                #     '<br />'.join('{}: {}'.format(k, v) for k, v in (node.parameters or {}).items()))
                sankey_node["label"].append(
                    "{}<br />".format(node.name)
                    + "<br />".join(
                        "{}: {}".format(k, v if len(str(v)) < 24 else (str(v)[:24] + " ..."))
                        for k, v in (node.parameters or {}).items()
                    )
                )

                sankey_node["color"].append(self._get_node_color(node))

                for p in parents:
                    sankey_link["source"].append(p)
                    sankey_link["target"].append(idx)
                    sankey_link["value"].append(1)

            # if nothing changed, we give up
            if nodes == next_nodes:
                break

            nodes = next_nodes

        # make sure we have no independent (unconnected) nodes
        single_nodes = []
        for i in [n for n in range(len(visited)) if n not in sankey_link["source"] and n not in sankey_link["target"]]:
            single_nodes.append(i)

        # create the sankey graph
        dag_flow = dict(
            link=sankey_link,
            node=sankey_node,
            textfont=dict(color="rgba(0,0,0,0)", size=1),
            type="sankey",
            orientation="h",
        )

        table_values = self._build_table_report(node_params, visited)

        # hack, show single node sankey
        if single_nodes:
            singles_flow = dict(
                x=list(range(len(single_nodes))),
                y=[1] * len(single_nodes),
                text=[v for i, v in enumerate(sankey_node["label"]) if i in single_nodes],
                mode="markers",
                hovertemplate="%{text}<extra></extra>",
                marker=dict(
                    color=[v for i, v in enumerate(sankey_node["color"]) if i in single_nodes],
                    size=[40] * len(single_nodes),
                ),
                showlegend=False,
                type="scatter",
            )
            # only single nodes
            if len(single_nodes) == len(sankey_node["label"]):
                fig = dict(
                    data=[singles_flow],
                    layout={
                        "hovermode": "closest",
                        "xaxis": {"visible": False},
                        "yaxis": {"visible": False},
                    },
                )
            else:
                dag_flow["domain"] = {"x": [0.0, 1.0], "y": [0.2, 1.0]}
                fig = dict(
                    data=[dag_flow, singles_flow],
                    layout={
                        "autosize": True,
                        "hovermode": "closest",
                        "xaxis": {
                            "anchor": "y",
                            "domain": [0.0, 1.0],
                            "visible": False,
                        },
                        "yaxis": {
                            "anchor": "x",
                            "domain": [0.0, 0.15],
                            "visible": False,
                        },
                    },
                )
        else:
            # create the sankey plot
            fig = dict(
                data=[dag_flow],
                layout={"xaxis": {"visible": False}, "yaxis": {"visible": False}},
            )

        # report DAG
        self._task.get_logger().report_plotly(
            title=self._report_plot_execution_flow["title"],
            series=self._report_plot_execution_flow["series"],
            iteration=0,
            figure=fig,
        )
        # report detailed table
        self._task.get_logger().report_table(
            title=self._report_plot_execution_details["title"],
            series=self._report_plot_execution_details["series"],
            iteration=0,
            table_plot=table_values,
        )

    def _build_table_report(self, node_params: List, visited: List) -> List[List]:
        """
        Create the detailed table report on all the jobs in the pipeline

        :param node_params: list of node parameters
        :param visited: list of nodes

        :return: Table as a List of a List of strings (cell)
        """
        task_link_template = (
            self._task.get_output_log_web_page()
            .replace("/{}/".format(self._task.project), "/{project}/")
            .replace("/{}/".format(self._task.id), "/{task}/")
        )

        table_values = [["Pipeline Step", "Task ID", "Task Name", "Status", "Parameters"]]

        for name, param in zip(visited, node_params):
            param_str = str(param) if param else ""
            if len(param_str) > 3:
                # remove {} from string
                param_str = param_str[1:-1]

            step_name = name
            if self._nodes[name].base_task_id:
                step_name += '\n[<a href="{}"> {} </a>]'.format(
                    task_link_template.format(project="*", task=self._nodes[name].base_task_id),
                    "base task",
                )

            table_values.append(
                [
                    step_name,
                    self.__create_task_link(self._nodes[name], task_link_template),
                    self._nodes[name].job.task.name if self._nodes[name].job else "",
                    str(self._nodes[name].status or ""),
                    param_str,
                ]
            )

        return table_values

    def _call_retries_callback(self, node: "PipelineController.Node") -> bool:
        # if this functions returns True, we should relaunch the node
        # if False, don't relaunch
        if node.name not in self._retries_callbacks:
            return False
        try:
            return self._retries_callbacks[node.name](self, node, self._retries.get(node.name, 0))
        except Exception as e:
            getLogger("clearml.automation.controller").warning(
                "Failed calling the retry callback for node '{}'. Error is '{}'".format(node.name, e)
            )
            return False

    @classmethod
    def _get_node_color(cls, node: "PipelineController.Node") -> str:
        # type (self.Mode) -> str
        """
        Return the node color based on the node/job state
        :param node: A node in the pipeline
        :return: string representing the color of the node (e.g. "red", "green", etc)
        """
        if not node:
            return ""

        color_lookup = {
            "failed": "red",
            "cached": "darkslateblue",
            "completed": "blue",
            "aborted": "royalblue",
            "queued": "#bdf5bd",
            "running": "green",
            "skipped": "gray",
            "pending": "lightsteelblue",
        }
        return color_lookup.get(node.status, "")

    def _update_nodes_status(self) -> None:
        # type () -> ()
        """
        Update the status of all nodes in the pipeline
        """
        jobs = []
        previous_status_map = {}
        # copy to avoid race condition
        nodes = self._nodes.copy()
        for name, node in nodes.items():
            if not node.job:
                continue
            # noinspection PyProtectedMember
            previous_status_map[name] = node.job._last_status
            jobs.append(node.job)
        BaseJob.update_status_batch(jobs)
        for node in nodes.values():
            self._update_node_status(node)

    def _update_node_status(self, node: "PipelineController.Node") -> None:
        # type (self.Node) -> ()
        """
        Update the node status entry based on the node/job state
        :param node: A node in the pipeline
        """
        previous_status = node.status

        if node.job and node.job.is_running():
            node.set_job_started()
        update_job_ended = node.job_started and not node.job_ended

        if node.executed is not None:
            if node.job and node.job.is_failed():
                # failed job
                node.status = "failed"
            elif node.job and node.job.is_cached_task():
                # cached job
                node.status = "cached"
            elif not node.job or node.job.is_completed():
                # completed job
                node.status = "completed"
            else:
                # aborted job
                node.status = "aborted"
        elif node.job:
            if node.job.is_pending():
                # lightgreen, pending in queue
                node.status = "queued"
            elif node.job.is_completed():
                # completed job
                node.status = "completed"
            elif node.job.is_failed():
                # failed job
                node.status = "failed"
            elif node.job.is_stopped():
                # aborted job
                node.status = "aborted"
            else:
                node.status = "running"
        elif node.skip_job:
            node.status = "skipped"
        else:
            node.status = "pending"

        if update_job_ended and node.status in ("aborted", "failed", "completed"):
            node.set_job_ended()

        if (
            previous_status is not None
            and previous_status != node.status
            and self._status_change_callbacks.get(node.name)
        ):
            # noinspection PyBroadException
            try:
                self._status_change_callbacks[node.name](self, node, previous_status)
            except Exception as e:
                getLogger("clearml.automation.controller").warning(
                    "Failed calling the status change callback for node '{}'. Error is '{}'".format(node.name, e)
                )

    def _update_dag_state_artifact(self) -> ():
        pipeline_dag = self._serialize()
        self._task.upload_artifact(
            name=self._state_artifact_name,
            artifact_object="",
            metadata=dict(pipeline=hash_dict(pipeline_dag)),
            preview=json.dumps(pipeline_dag, indent=1),
        )

    def _force_task_configuration_update(self) -> ():
        pipeline_dag = self._serialize()
        if self._task:
            # noinspection PyProtectedMember
            self._task._set_configuration(
                name=self._config_section,
                config_type="dictionary",
                description="pipeline state: {}".format(hash_dict(pipeline_dag)),
                config_text=json.dumps(pipeline_dag, indent=2),
                force=True,
            )

    def _update_progress(self) -> ():
        """
        Update progress of the pipeline every PipelineController._update_progress_interval seconds.
        Progress is calculated as the mean of the progress of each step in the pipeline.
        """
        if time() - self._last_progress_update_time < self._update_progress_interval:
            return
        # copy to avoid race condition
        nodes = self._nodes.copy()
        job_progress = [(node.job.task.get_progress() or 0) if node.job else 0 for node in nodes.values()]
        if len(job_progress):
            self._task.set_progress(int(sum(job_progress) / len(job_progress)))
        self._last_progress_update_time = time()

    def _daemon(self) -> ():
        """
        The main pipeline execution loop. This loop is executed on its own dedicated thread.
        :return:
        """
        launch_thread_pool = ThreadPool(16)
        pooling_counter = 0
        launched_nodes = set()
        last_monitor_report = last_plot_report = time()
        while self._stop_event:
            # stop request
            if self._stop_event.wait(self._pool_frequency if pooling_counter else 0.01):
                break

            pooling_counter += 1

            # check the pipeline time limit
            if self._pipeline_time_limit and (time() - self._start_time) > self._pipeline_time_limit:
                break

            self._update_progress()
            self._update_nodes_status()
            # check the state of all current jobs
            # if no a job ended, continue
            completed_jobs = []
            force_execution_plot_update = False
            nodes_failed_stop_pipeline = []
            for j in self._running_nodes:
                node = self._nodes[j]
                if not node.job:
                    continue
                if node.job.is_stopped(aborted_nonresponsive_as_running=True):
                    node_failed = node.job.is_failed()
                    if node_failed:
                        if self._call_retries_callback(node):
                            self._relaunch_node(node)
                            continue
                        else:
                            self._final_failure[node.name] = True

                    completed_jobs.append(j)
                    if node.job.is_aborted():
                        node.executed = node.job.task_id() if not node.skip_children_on_abort else False
                    elif node_failed:
                        node.executed = node.job.task_id() if not node.skip_children_on_fail else False
                    else:
                        node.executed = node.job.task_id()

                    if j in launched_nodes:
                        launched_nodes.remove(j)
                    # check if we need to stop all running steps
                    if node_failed and self._abort_running_steps_on_failure and not node.continue_on_fail:
                        nodes_failed_stop_pipeline.append(node.name)
                elif node.timeout:
                    started = node.job.task.data.started
                    if (datetime.now().astimezone(started.tzinfo) - started).total_seconds() > node.timeout:
                        node.job.abort()
                        completed_jobs.append(j)
                        node.executed = node.job.task_id()
                elif j in launched_nodes and node.job.is_running():
                    # make sure update the execution graph when the job started running
                    # (otherwise it will still be marked queued)
                    launched_nodes.remove(j)
                    force_execution_plot_update = True

            # update running jobs
            self._running_nodes = [j for j in self._running_nodes if j not in completed_jobs]

            # nothing changed, we can sleep
            if not completed_jobs and self._running_nodes:
                # force updating the pipeline state (plot) at least every 5 min.
                if force_execution_plot_update or time() - last_plot_report > self._update_execution_plot_interval:
                    last_plot_report = time()
                    last_monitor_report = time()
                    self.update_execution_plot()
                elif time() - last_monitor_report > self._monitor_node_interval:
                    last_monitor_report = time()
                    self._scan_monitored_nodes()
                continue

            # callback on completed jobs
            if self._experiment_completed_cb or self._post_step_callbacks:
                for job in completed_jobs:
                    job_node = self._nodes.get(job)
                    if not job_node:
                        continue
                    if self._experiment_completed_cb:
                        self._experiment_completed_cb(self, job_node)
                    if self._post_step_callbacks.get(job_node.name):
                        self._post_step_callbacks[job_node.name](self, job_node)

            # check if we need to stop the pipeline, and abort all running steps
            if nodes_failed_stop_pipeline:
                print(
                    "Aborting pipeline and stopping all running steps, node {} failed".format(
                        nodes_failed_stop_pipeline
                    )
                )
                break

            # Pull the next jobs in the pipeline, based on the completed list
            next_nodes = []
            for node in list(self._nodes.values()):
                # check if already processed or needs to be skipped
                if node.job or node.executed or node.skip_job:
                    continue
                completed_parents = [bool(p in self._nodes and self._nodes[p].executed) for p in node.parents or []]
                if all(completed_parents):
                    next_nodes.append(node.name)

            # update the execution graph
            print("Launching the next {} steps".format(len(next_nodes)))
            node_launch_success = launch_thread_pool.map(self._launch_node, [self._nodes[name] for name in next_nodes])
            for name, success in zip(next_nodes, node_launch_success):
                if success and not self._nodes[name].skip_job:
                    if self._nodes[name].job and self._nodes[name].job.task_parameter_override is not None:
                        self._nodes[name].job.task_parameter_override.update(self._nodes[name].parameters or {})
                    print("Launching step: {}".format(name))
                    print(
                        "Parameters:\n{}".format(
                            self._nodes[name].job.task_parameter_override
                            if self._nodes[name].job
                            else self._nodes[name].parameters
                        )
                    )
                    print("Configurations:\n{}".format(self._nodes[name].configurations))
                    print("Overrides:\n{}".format(self._nodes[name].task_overrides))
                    launched_nodes.add(name)
                    # check if node is cached do not wait for event but run the loop again
                    if self._nodes[name].executed:
                        pooling_counter = 0
                else:
                    getLogger("clearml.automation.controller").warning(
                        "Skipping launching step '{}': {}".format(name, self._nodes[name])
                    )

            # update current state (in configuration, so that we could later continue an aborted pipeline)
            # visualize pipeline state (plot)
            self.update_execution_plot()

            # quit if all pipelines nodes are fully executed.
            if not next_nodes and not self._running_nodes:
                break

        # stop all currently running jobs:
        for node in list(self._nodes.values()):
            if node.executed is False and not node.continue_on_fail:
                self._pipeline_task_status_failed = True

            if node.job and not node.job.is_stopped():
                node.job.abort()
            elif not node.job and not node.executed:
                # mark Node as skipped if it has no Job object and it is not executed
                node.skip_job = True

        # visualize pipeline state (plot)
        self.update_execution_plot()

        if self._stop_event:
            # noinspection PyBroadException
            try:
                self._stop_event.set()
            except Exception:
                pass

    def _parse_step_ref(self, value: Any, recursive: bool = False) -> Optional[str]:
        """
        Return the step reference. For example ``"${step1.parameters.Args/param}"``
        :param value: string
        :param recursive: if True, recursively parse all values in the dict, list or tuple
        :return:
        """
        # look for all the step references
        pattern = self._step_ref_pattern
        updated_value = value
        if isinstance(value, str):
            for g in pattern.findall(value):
                # update with actual value
                new_val = self.__parse_step_reference(g)
                if not isinstance(new_val, six.string_types):
                    return new_val
                updated_value = updated_value.replace(g, new_val, 1)

        # if we have a dict, list or tuple, we need to recursively update the values
        if recursive:
            if isinstance(value, dict):
                updated_value = {}
                for k, v in value.items():
                    updated_value[k] = self._parse_step_ref(v, recursive=True)
            elif isinstance(value, list):
                updated_value = [self._parse_step_ref(v, recursive=True) for v in value]
            elif isinstance(value, tuple):
                updated_value = tuple(self._parse_step_ref(v, recursive=True) for v in value)

        return updated_value

    def _parse_task_overrides(self, task_overrides: dict) -> dict:
        """
        Return the step reference. For example ``"${step1.parameters.Args/param}"``
        :param task_overrides: string
        :return:
        """
        updated_overrides = {}
        for k, v in task_overrides.items():
            updated_overrides[k] = self._parse_step_ref(v)

        return updated_overrides

    def _verify_node_name(self, name: str) -> None:
        if name in self._nodes:
            raise ValueError("Node named '{}' already exists in the pipeline dag".format(name))
        if name in self._reserved_pipeline_names:
            raise ValueError("Node named '{}' is a reserved keyword, use a different name".format(name))

    def _scan_monitored_nodes(self) -> None:
        """
        Scan all nodes and monitor their metrics/artifacts/models
        """
        for node in list(self._nodes.values()):
            self._monitor_node(node)

    def _monitor_node(self, node: "PipelineController.Node") -> None:
        """
        If Node is running, put the metrics from the node on the pipeline itself.
        :param node: Node to test
        """
        if not node:
            return

        # verify we have the node
        if node.name not in self._monitored_nodes:
            self._monitored_nodes[node.name] = {}

        # if we are done with this node, skip it
        if self._monitored_nodes[node.name].get("completed"):
            return

        if node.job and node.job.task:
            task = node.job.task
        elif node.job and node.executed and isinstance(node.executed, str):
            task = Task.get_task(task_id=node.executed)
        else:
            return

        # update the metrics
        if node.monitor_metrics:
            metrics_state = self._monitored_nodes[node.name].get("metrics", {})
            logger = self._task.get_logger()
            scalars = task.get_reported_scalars(x_axis="iter")
            for (s_title, s_series), (t_title, t_series) in node.monitor_metrics:
                values = scalars.get(s_title, {}).get(s_series)
                if values and values.get("x") is not None and values.get("y") is not None:
                    x = values["x"][-1]
                    y = values["y"][-1]
                    last_y = metrics_state.get(s_title, {}).get(s_series)
                    if last_y is None or y > last_y:
                        logger.report_scalar(title=t_title, series=t_series, value=y, iteration=int(x))
                        last_y = y
                    if not metrics_state.get(s_title):
                        metrics_state[s_title] = {}
                    metrics_state[s_title][s_series] = last_y

            self._monitored_nodes[node.name]["metrics"] = metrics_state

        if node.monitor_artifacts:
            task.reload()
            artifacts = task.data.execution.artifacts
            self._task.reload()
            output_artifacts = []
            for s_artifact, t_artifact in node.monitor_artifacts:
                # find artifact
                for a in artifacts:
                    if a.key != s_artifact:
                        continue

                    new_a = copy(a)
                    new_a.key = t_artifact
                    output_artifacts.append(new_a)
                    break

            # update artifacts directly on the Task
            if output_artifacts:
                # noinspection PyProtectedMember
                self._task._add_artifacts(output_artifacts)

        if node.monitor_models:
            task.reload()
            output_models = task.data.models.output
            self._task.reload()
            target_models = []
            for s_model, t_model in node.monitor_models:
                # find artifact
                for a in output_models:
                    if a.name != s_model:
                        continue

                    new_a = copy(a)
                    new_a.name = t_model
                    target_models.append(new_a)
                    break

            # update artifacts directly on the Task
            if target_models:
                self._task.reload()
                models = self._task.data.models
                keys = [a.name for a in target_models]
                models.output = [a for a in models.output or [] if a.name not in keys] + target_models
                # noinspection PyProtectedMember
                self._task._edit(models=models)

        # update the state (so that we do not scan the node twice)
        if node.job.is_stopped(aborted_nonresponsive_as_running=True):
            self._monitored_nodes[node.name]["completed"] = True

    def _get_target_project(self, return_project_id: bool = False) -> str:
        """
        return the pipeline components target folder name/id

        :param return_project_id: if False (default), return target folder name. If True, return project id

        :return: project id/name (None if not valid)
        """
        if not self._target_project:
            return ""

        if str(self._target_project).lower().strip() == "true":
            if not self._task:
                return ""
            return self._task.project if return_project_id else self._task.get_project_name()

        if not return_project_id:
            return self._target_project

        return get_or_create_project(
            session=self._task.session if self._task else Task.default_session,
            project_name=self._target_project,
        )

    @classmethod
    def _add_pipeline_name_run_number(cls, task: Task) -> None:
        if not task:
            return
        # if we were already executed, do not rename (meaning aborted pipeline that was continued)
        # noinspection PyProtectedMember
        if task._get_runtime_properties().get(cls._runtime_property_hash):
            return

        # remove the #<num> suffix if we have one:
        task_name = re.compile(r" #\d+$").split(task.name or "", 1)[0]
        page_size = 100
        # find exact name or " #<num>" extension
        prev_pipelines_ids = task.query_tasks(
            task_name=r"^{}(| #\d+)$".format(task_name),
            task_filter=dict(
                project=[task.project],
                system_tags=[cls._tag],
                order_by=["-created"],
                page_size=page_size,
                fetch_only_first_page=True,
            ),
        )
        max_value = len(prev_pipelines_ids) if prev_pipelines_ids else 0
        # we hit the limit
        if max_value == page_size:
            # make sure that if we get something wrong we do not stop the pipeline,
            # worst case fail to auto increment
            try:
                # we assume we are the latest so let's take a few (last 10) and check the max number
                last_task_name: List[Dict] = task.query_tasks(
                    task_filter=dict(task_ids=prev_pipelines_ids[:10], project=[task.project]),
                    additional_return_fields=["name"],
                )
                # let's parse the names
                pattern = re.compile(r" #(?P<key>\d+)$")
                task_parts = [pattern.split(t.get("name") or "", 1) for t in last_task_name]
                # find the highest number
                for parts in task_parts:
                    if len(parts) >= 2:
                        try:
                            max_value = max(max_value, int(parts[1]) + 1)
                        except (TypeError, ValueError):
                            pass
            except Exception as ex:
                getLogger("clearml.automation.controller").warning(
                    "Pipeline auto run increment failed (skipping): {}".format(ex)
                )
                max_value = 0

        if max_value > 1:
            task.set_name(task_name + " #{}".format(max_value))

    @classmethod
    def _get_pipeline_task(cls) -> Task:
        """
        Return the pipeline Task (either the current one, or the parent Task of the currently running Task)
        Raise ValueError if we could not locate the pipeline Task

        :return: Pipeline Task
        """
        # get main Task.
        task = Task.current_task()
        if str(task.task_type) == str(Task.TaskTypes.controller) and cls._tag in task.get_system_tags():
            return task
        # get the parent Task, it should be the pipeline
        if not task.parent:
            raise ValueError("Could not locate parent Pipeline Task")
        parent = Task.get_task(task_id=task.parent)
        if str(parent.task_type) == str(Task.TaskTypes.controller) and cls._tag in parent.get_system_tags():
            return parent
        raise ValueError("Could not locate parent Pipeline Task")

    def __verify_step_reference(self, node: "PipelineController.Node", step_ref_string: str) -> Optional[str]:
        """
        Verify the step reference. For example ``"${step1.parameters.Args/param}"``
        Raise ValueError on misconfiguration

        :param Node node: calling reference node (used for logging)
        :param str step_ref_string: For example ``"${step1.parameters.Args/param}"``
        :return: If step reference is used, return the pipeline step name, otherwise return None
        """
        parts = step_ref_string[2:-1].split(".")
        v = step_ref_string
        if len(parts) < 2:
            raise ValueError("Node '{}', parameter '{}' is invalid".format(node.name, v))
        prev_step = parts[0]
        input_type = parts[1]

        # check if we reference the pipeline arguments themselves
        if prev_step == self._pipeline_step_ref:
            if input_type not in self._pipeline_args:
                raise ValueError("Node '{}', parameter '{}', step name '{}' is invalid".format(node.name, v, prev_step))
            return None

        if prev_step not in self._nodes:
            raise ValueError("Node '{}', parameter '{}', step name '{}' is invalid".format(node.name, v, prev_step))
        if input_type not in ("artifacts", "parameters", "models", "id"):
            raise ValueError("Node {}, parameter '{}', input type '{}' is invalid".format(node.name, v, input_type))

        if input_type != "id" and len(parts) < 3:
            raise ValueError("Node '{}', parameter '{}' is invalid".format(node.name, v))

        if input_type == "models":
            try:
                model_type = parts[2].lower()
            except Exception:
                raise ValueError(
                    "Node '{}', parameter '{}', input type '{}', model_type is missing {}".format(
                        node.name, v, input_type, parts
                    )
                )
            if model_type not in ("input", "output"):
                raise ValueError(
                    "Node '{}', parameter '{}', input type '{}', "
                    "model_type is invalid (input/output) found {}".format(node.name, v, input_type, model_type)
                )

            if len(parts) < 4:
                raise ValueError(
                    "Node '{}', parameter '{}', input type '{}', model index is missing".format(
                        node.name, v, input_type
                    )
                )

            # check casting
            try:
                int(parts[3])
            except Exception:
                raise ValueError(
                    "Node '{}', parameter '{}', input type '{}', model index is missing {}".format(
                        node.name, v, input_type, parts
                    )
                )

            if len(parts) < 5:
                raise ValueError(
                    "Node '{}', parameter '{}', input type '{}', model property is missing".format(
                        node.name, v, input_type
                    )
                )

            if not hasattr(BaseModel, parts[4]):
                raise ValueError(
                    "Node '{}', parameter '{}', input type '{}', model property is invalid {}".format(
                        node.name, v, input_type, parts[4]
                    )
                )
        return prev_step

    def __parse_step_reference(self, step_ref_string: str) -> str:
        """
        return the adjusted value for "${step...}"
        :param step_ref_string: reference string of the form ${step_name.type.value}"
        :return: str with value
        """
        parts = step_ref_string[2:-1].split(".")
        if len(parts) < 2:
            raise ValueError("Could not parse reference '{}'".format(step_ref_string))
        prev_step = parts[0]
        input_type = parts[1].lower()

        # check if we reference the pipeline arguments themselves
        if prev_step == self._pipeline_step_ref:
            if parts[1] not in self._pipeline_args:
                raise ValueError(
                    "Could not parse reference '{}', "
                    "pipeline argument '{}' could not be found".format(step_ref_string, parts[1])
                )
            return self._pipeline_args[parts[1]]

        if prev_step not in self._nodes or (
            not self._nodes[prev_step].job
            and not self._nodes[prev_step].executed
            and not self._nodes[prev_step].base_task_id
        ):
            raise ValueError(
                "Could not parse reference '{}', step '{}' could not be found".format(step_ref_string, prev_step)
            )

        if input_type not in (
            "artifacts",
            "parameters",
            "models",
            "id",
            "script",
            "execution",
            "container",
            "output",
            "comment",
            "models",
            "tags",
            "system_tags",
            "project",
        ):
            raise ValueError("Could not parse reference '{}', type '{}' not valid".format(step_ref_string, input_type))
        if input_type != "id" and len(parts) < 3:
            raise ValueError("Could not parse reference '{}', missing fields in '{}'".format(step_ref_string, parts))

        task = (
            self._nodes[prev_step].job.task
            if self._nodes[prev_step].job
            else Task.get_task(task_id=self._nodes[prev_step].executed or self._nodes[prev_step].base_task_id)
        )
        task.reload()
        if input_type == "artifacts":
            # fix \. to use . in artifacts
            artifact_path = (".".join(parts[2:])).replace("\\.", "\\_dot_\\")
            artifact_path = artifact_path.split(".")

            obj = task.artifacts
            for p in artifact_path:
                p = p.replace("\\_dot_\\", ".")
                if isinstance(obj, dict):
                    obj = obj.get(p)
                elif hasattr(obj, p):
                    obj = getattr(obj, p)
                else:
                    raise ValueError(
                        "Could not locate artifact {} on previous step {}".format(".".join(parts[1:]), prev_step)
                    )
            return str(obj)
        elif input_type == "parameters":
            step_params = task.get_parameters()
            param_name = ".".join(parts[2:])
            if param_name not in step_params:
                raise ValueError(
                    "Could not locate parameter {} on previous step {}".format(".".join(parts[1:]), prev_step)
                )
            return step_params.get(param_name)
        elif input_type == "models":
            model_type = parts[2].lower()
            if model_type not in ("input", "output"):
                raise ValueError("Could not locate model {} on previous step {}".format(".".join(parts[1:]), prev_step))
            try:
                model_idx = int(parts[3])
                model = task.models[model_type][model_idx]
            except Exception:
                raise ValueError(
                    "Could not locate model {} on previous step {}, index {} is invalid".format(
                        ".".join(parts[1:]), prev_step, parts[3]
                    )
                )

            return str(getattr(model, parts[4]))
        elif input_type == "id":
            return task.id
        elif input_type in (
            "script",
            "execution",
            "container",
            "output",
            "comment",
            "models",
            "tags",
            "system_tags",
            "project",
        ):
            # noinspection PyProtectedMember
            return task._get_task_property(".".join(parts[1:]))

        return None

    @classmethod
    def __create_task_link(cls, a_node: "PipelineController.Node", task_link_template: str) -> str:
        if not a_node:
            return ""
        # create the detailed parameter table
        task_id = project_id = None
        if a_node.job:
            project_id = a_node.job.task.project
            task_id = a_node.job.task.id
        elif a_node.executed:
            task_id = a_node.executed
            if cls._task_project_lookup.get(task_id):
                project_id = cls._task_project_lookup[task_id]
            else:
                # noinspection PyBroadException
                try:
                    project_id = Task.get_task(task_id=task_id).project
                except Exception:
                    project_id = "*"
                cls._task_project_lookup[task_id] = project_id

        if not task_id:
            return ""

        return '<a href="{}"> {} </a>'.format(task_link_template.format(project=project_id, task=task_id), task_id)

    def _default_retry_on_failure_callback(
        self,
        _pipeline_controller: "PipelineController",
        _node: "PipelineController.Node",
        retries: int,
        max_retries: Optional[int] = None,
    ) -> bool:
        return retries < (self._def_max_retry_on_failure if max_retries is None else max_retries)

    def _upload_pipeline_artifact(self, artifact_name: str, artifact_object: Any) -> None:
        self._task.upload_artifact(
            name=artifact_name,
            artifact_object=artifact_object,
            wait_on_upload=True,
            extension_name=(
                ".pkl" if isinstance(artifact_object, dict) and not self._artifact_serialization_function else None
            ),
            serialization_function=self._artifact_serialization_function,
        )


class PipelineDecorator(PipelineController):
    _added_decorator: List[dict] = []
    _ref_lazy_loader_id_to_node_name: dict = {}
    _singleton: Optional["PipelineDecorator"] = None
    _eager_step_artifact = "eager_step"
    _eager_execution_instance = False
    _debug_execute_step_process = False
    _debug_execute_step_function = False
    _default_execution_queue = None
    _multi_pipeline_instances = []
    _multi_pipeline_call_counter = -1
    _atexit_registered = False

    def __init__(
        self,
        name: str,
        project: str,
        version: Optional[str] = None,
        pool_frequency: float = 0.2,
        add_pipeline_tags: bool = False,
        target_project: Optional[str] = None,
        abort_on_failure: bool = False,
        add_run_number: bool = True,
        retry_on_failure: Optional[
            Union[int, Callable[[PipelineController, PipelineController.Node, int], bool]]
        ] = None,  # noqa
        docker: Optional[str] = None,
        docker_args: Optional[str] = None,
        docker_bash_setup_script: Optional[str] = None,
        packages: Optional[Union[bool, str, Sequence[str]]] = None,
        repo: Optional[str] = None,
        repo_branch: Optional[str] = None,
        repo_commit: Optional[str] = None,
        artifact_serialization_function: Optional[Callable[[Any], Union[bytes, bytearray]]] = None,
        artifact_deserialization_function: Optional[Callable[[bytes], Any]] = None,
        output_uri: Optional[Union[str, bool]] = None,
        skip_global_imports: bool = False,
        working_dir: Optional[str] = None,
        enable_local_imports: bool = True,
    ) -> ():
        """
        Create a new pipeline controller. The newly created object will launch and monitor the new experiments.

        :param name: Provide pipeline name (if main Task exists it overrides its name)
        :param project: Provide project storing the pipeline (if main Task exists  it overrides its project)
        :param version: Pipeline version. This version allows to uniquely identify the pipeline
            template execution. Examples for semantic versions: version='1.0.1' , version='23', version='1.2'.
            If not set, find the latest version of the pipeline and increment it. If no such version is found,
            default to '1.0.0'
        :param float pool_frequency: The pooling frequency (in minutes) for monitoring experiments / states.
        :param bool add_pipeline_tags: (default: False) if True, add `pipe: <pipeline_task_id>` tag to all
            steps (Tasks) created by this pipeline.
        :param str target_project: If provided, all pipeline steps are cloned into the target project
        :param bool abort_on_failure: If False (default), failed pipeline steps will not cause the pipeline
            to stop immediately, instead any step that is not connected (or indirectly connected) to the failed step,
            will still be executed. Nonetheless, the pipeline itself will be marked failed, unless the failed step
            was specifically defined with "continue_on_fail=True".
            If True, any failed step will cause the pipeline to immediately abort, stop all running steps,
            and mark the pipeline as failed.
        :param add_run_number: If True (default), add the run number of the pipeline to the pipeline name.
            Example, the second time we launch the pipeline "best pipeline", we rename it to "best pipeline #2"
        :param retry_on_failure: Integer (number of retries) or Callback function that returns True to allow a retry

          - Integer: In case of node failure, retry the node the number of times indicated by this parameter.
          - Callable: A function called on node failure. Takes as parameters:
            the PipelineController instance, the PipelineController.Node that failed and an int
            representing the number of previous retries for the node that failed.
            The function must return ``True`` if the node should be retried and ``False`` otherwise.
            If True, the node will be re-queued and the number of retries left will be decremented by 1.
            By default, if this callback is not specified, the function will be retried the number of
            times indicated by `retry_on_failure`.

              .. code-block:: py

                  def example_retry_on_failure_callback(pipeline, node, retries):
                      print(node.name, ' failed')
                      # allow up to 5 retries (total of 6 runs)
                      return retries < 5
        :param docker: Select the docker image to be executed in by the remote session
        :param docker_args: Add docker arguments, pass a single string
        :param docker_bash_setup_script: Add bash script to be executed
            inside the docker before setting up the Task's environment
        :param packages: Manually specify a list of required packages or a local requirements.txt file.
            Example: ["tqdm>=2.1", "scikit-learn"] or "./requirements.txt"
            If not provided, packages are automatically added.
            Use `False` to install requirements from "requirements.txt" inside your git repository
        :param repo: Optional, specify a repository to attach to the pipeline controller, when remotely executing.
            Allow users to execute the controller inside the specified repository, enabling them to load modules/script
            from the repository. Notice the execution work directory will be the repository root folder.
            Supports both git repo url link, and local repository path (automatically converted into the remote
            git/commit as is currently checkout).
            Example remote url: 'https://github.com/user/repo.git'
            Example local repo copy: './repo' -> will automatically store the remote
            repo url and commit ID based on the locally cloned copy
            Use empty string ("") to disable any repository auto-detection
        :param repo_branch: Optional, specify the remote repository branch (Ignored, if local repo path is used)
        :param repo_commit: Optional, specify the repository commit ID (Ignored, if local repo path is used)
        :param artifact_serialization_function: A serialization function that takes one
            parameter of any type which is the object to be serialized. The function should return
            a `bytes` or `bytearray` object, which represents the serialized object. All parameter/return
            artifacts uploaded by the pipeline will be serialized using this function.
            All relevant imports must be done in this function. For example:

            .. code-block:: py

                def serialize(obj):
                    import dill
                    return dill.dumps(obj)
        :param artifact_deserialization_function: A deserialization function that takes one parameter of type `bytes`,
            which represents the serialized object. This function should return the deserialized object.
            All parameter/return artifacts fetched by the pipeline will be deserialized using this function.
            All relevant imports must be done in this function. For example:

            .. code-block:: py

                def deserialize(bytes_):
                    import dill
                    return dill.loads(bytes_)
        :param output_uri: The storage / output url for this pipeline. This is the default location for output
            models and other artifacts. Check Task.init reference docs for more info (output_uri is a parameter).
            The `output_uri` of this pipeline's steps will default to this value.
        :param skip_global_imports: If True, global imports will not be included in the steps' execution, otherwise all
            global imports will be automatically imported in a safe manner at the beginning of each step’s execution.
            Default is False
        :param working_dir: Working directory to launch the pipeline from.
        :param enable_local_imports: If True, allow pipeline steps to import from local files
            by appending to the PYTHONPATH of each step the directory the pipeline controller
            script resides in (sys.path[0]).
            If False, the directory won't be appended to PYTHONPATH. Default is True.
            Ignored while running remotely.
        """
        super(PipelineDecorator, self).__init__(
            name=name,
            project=project,
            version=version,
            pool_frequency=pool_frequency,
            add_pipeline_tags=add_pipeline_tags,
            target_project=target_project,
            abort_on_failure=abort_on_failure,
            add_run_number=add_run_number,
            retry_on_failure=retry_on_failure,
            docker=docker,
            docker_args=docker_args,
            docker_bash_setup_script=docker_bash_setup_script,
            packages=packages,
            repo=repo,
            repo_branch=repo_branch,
            repo_commit=repo_commit,
            always_create_from_code=False,
            artifact_serialization_function=artifact_serialization_function,
            artifact_deserialization_function=artifact_deserialization_function,
            output_uri=output_uri,
            skip_global_imports=skip_global_imports,
            working_dir=working_dir,
            enable_local_imports=enable_local_imports,
        )
        # if we are in eager execution, make sure parent class knows it
        if self._eager_execution_instance:
            self._mock_execution = True

        if PipelineDecorator._default_execution_queue:
            super(PipelineDecorator, self).set_default_execution_queue(PipelineDecorator._default_execution_queue)

        for n in self._added_decorator:
            self._add_function_step(**n)
        self._added_decorator.clear()
        PipelineDecorator._singleton = self
        self._reference_callback = []
        # store launched nodes, in case we call the same function multiple times, and need renaming:
        self._launched_step_names = set()
        # map eager steps task id to the new step name
        self._eager_steps_task_id: Dict[str, str] = {}

    def _daemon(self) -> ():
        """
        The main pipeline execution loop. This loop is executed on its own dedicated thread.
        override the daemon function, we only need to update the state

        :return:
        """
        pooling_counter = 0
        launched_nodes = set()
        last_monitor_report = last_plot_report = time()
        while self._stop_event:
            # stop request
            if self._stop_event.wait(self._pool_frequency if pooling_counter else 0.01):
                break

            pooling_counter += 1

            # check the pipeline time limit
            if self._pipeline_time_limit and (time() - self._start_time) > self._pipeline_time_limit:
                break

            self._update_progress()
            self._update_nodes_status()
            # check the state of all current jobs
            # if no a job ended, continue
            completed_jobs = []
            nodes_failed_stop_pipeline = []
            force_execution_plot_update = False
            for j in self._running_nodes:
                node = self._nodes[j]
                if not node.job:
                    continue
                if node.job.is_stopped(aborted_nonresponsive_as_running=True):
                    node_failed = node.job.is_failed()
                    if node_failed:
                        if self._call_retries_callback(node):
                            self._relaunch_node(node)
                            continue
                        else:
                            self._final_failure[node.name] = True
                    completed_jobs.append(j)

                    if node.job.is_aborted():
                        node.executed = node.job.task_id() if not node.skip_children_on_abort else False
                    elif node_failed:
                        node.executed = node.job.task_id() if not node.skip_children_on_fail else False
                    else:
                        node.executed = node.job.task_id()

                    if j in launched_nodes:
                        launched_nodes.remove(j)
                    # check if we need to stop all running steps
                    if node_failed and self._abort_running_steps_on_failure and not node.continue_on_fail:
                        nodes_failed_stop_pipeline.append(node.name)
                elif node.timeout:
                    started = node.job.task.data.started
                    if (datetime.now().astimezone(started.tzinfo) - started).total_seconds() > node.timeout:
                        node.job.abort()
                        completed_jobs.append(j)
                        node.executed = node.job.task_id()
                elif j in launched_nodes and node.job.is_running():
                    # make sure update the execution graph when the job started running
                    # (otherwise it will still be marked queued)
                    launched_nodes.remove(j)
                    force_execution_plot_update = True

            # update running jobs
            self._running_nodes = [j for j in self._running_nodes if j not in completed_jobs]

            # nothing changed, we can sleep
            if not completed_jobs and self._running_nodes:
                # force updating the pipeline state (plot) at least every 5 min.
                if force_execution_plot_update or time() - last_plot_report > self._update_execution_plot_interval:
                    last_plot_report = time()
                    last_monitor_report = time()
                    self.update_execution_plot()
                elif time() - last_monitor_report > self._monitor_node_interval:
                    last_monitor_report = time()
                    self._scan_monitored_nodes()
                continue

            # callback on completed jobs
            if self._experiment_completed_cb or self._post_step_callbacks:
                for job in completed_jobs:
                    job_node = self._nodes.get(job)
                    if not job_node:
                        continue
                    if self._experiment_completed_cb:
                        self._experiment_completed_cb(self, job_node)
                    if self._post_step_callbacks.get(job_node.name):
                        self._post_step_callbacks[job_node.name](self, job_node)

            # check if we need to stop the pipeline, and abort all running steps
            if nodes_failed_stop_pipeline:
                print(
                    "Aborting pipeline and stopping all running steps, node {} failed".format(
                        nodes_failed_stop_pipeline
                    )
                )
                break

            # update current state (in configuration, so that we could later continue an aborted pipeline)
            self._force_task_configuration_update()

            # visualize pipeline state (plot)
            self.update_execution_plot()

        # stop all currently running jobs, protect against changes while iterating):
        for node in list(self._nodes.values()):
            if node.executed is False and not node.continue_on_fail:
                self._pipeline_task_status_failed = True

            if node.job and not node.job.is_stopped():
                node.job.abort()
            elif not node.job and not node.executed:
                # mark Node as skipped if it has no Job object and it is not executed
                node.skip_job = True
                # if this is a standalone node, we need to remove it from the graph
                if not node.parents:
                    # check if this node is anyone's parent
                    found_parent = False
                    for v in list(self._nodes.values()):
                        if node.name in (v.parents or []):
                            found_parent = True
                            break
                    if not found_parent:
                        self._nodes.pop(node.name, None)

        # visualize pipeline state (plot)
        self.update_execution_plot()
        self._scan_monitored_nodes()

        if self._stop_event:
            # noinspection PyBroadException
            try:
                self._stop_event.set()
            except Exception:
                pass

    def update_execution_plot(self) -> ():
        """
        Update sankey diagram of the current pipeline
        """
        with self._reporting_lock:
            self._update_eager_generated_steps()
            super(PipelineDecorator, self).update_execution_plot()

    def _update_eager_generated_steps(self) -> None:
        # noinspection PyProtectedMember
        self._task.reload()
        artifacts = self._task.data.execution.artifacts
        # check if we have a new step on the DAG
        eager_artifacts = []
        for a in artifacts:
            if a.key and a.key.startswith("{}:".format(self._eager_step_artifact)):
                # expected value: '"eager_step":"parent-node-task-id":"eager-step-task-id'
                eager_artifacts.append(a)

        # verify we have the step, if we do not, add it.
        delete_artifact_keys = []
        for artifact in eager_artifacts:
            _, parent_step_task_id, eager_step_task_id = artifact.key.split(":", 2)

            # deserialize node definition
            eager_node_def = json.loads(artifact.type_data.preview)
            eager_node_name, eager_node_def = list(eager_node_def.items())[0]

            # verify we do not have any new nodes on the DAG (i.e. a step generating a Node eagerly)
            parent_node = None
            for node in list(self._nodes.values()):
                if not node.job and not node.executed:
                    continue
                t_id = node.executed or node.job.task_id
                if t_id == parent_step_task_id:
                    parent_node = node
                    break

            if not parent_node:
                # should not happen
                continue

            new_step_node_name = "{}_{}".format(parent_node.name, eager_node_name)
            counter = 1
            while new_step_node_name in self._nodes:
                new_step_node_name = "{}_{}".format(new_step_node_name, counter)
                counter += 1

            eager_node_def["name"] = new_step_node_name
            eager_node_def["parents"] = [parent_node.name]
            is_cached = eager_node_def.pop("is_cached", None)
            self._nodes[new_step_node_name] = self.Node(**eager_node_def)
            self._nodes[new_step_node_name].job = RunningJob(existing_task=eager_step_task_id)
            if is_cached:
                self._nodes[new_step_node_name].job.force_set_is_cached(is_cached)

            # make sure we will not rescan it.
            delete_artifact_keys.append(artifact.key)

        # remove all processed eager step artifacts
        if delete_artifact_keys:
            # noinspection PyProtectedMember
            self._task._delete_artifacts(delete_artifact_keys)
            self._force_task_configuration_update()

    def _create_task_from_function(
        self,
        docker: Optional[str],
        docker_args: Optional[str],
        docker_bash_setup_script: Optional[str],
        function: Callable,
        function_input_artifacts: Dict[str, str],
        function_kwargs: Dict[str, Any],
        function_return: List[str],
        auto_connect_frameworks: Optional[dict],
        auto_connect_arg_parser: Optional[dict],
        packages: Optional[Union[bool, str, Sequence[str]]],
        project_name: Optional[str],
        task_name: Optional[str],
        task_type: Optional[str],
        repo: Optional[str],
        branch: Optional[str],
        commit: Optional[str],
        helper_functions: Optional[Sequence[Callable]],
        output_uri: Optional[Union[str, bool]] = None,
        working_dir: Optional[str] = None,
    ) -> dict:
        def sanitize(function_source: str) -> str:
            matched = re.match(r"[\s]*@[\w]*.component[\s\\]*\(", function_source)
            if matched:
                function_source = function_source[matched.span()[1] :]
                # find the last ")"
                open_parenthesis = 0
                last_index = -1
                for i, c in enumerate(function_source):
                    if not open_parenthesis and c == ")":
                        last_index = i
                        break
                    elif c == ")":
                        open_parenthesis -= 1
                    elif c == "(":
                        open_parenthesis += 1
                if last_index >= 0:
                    function_source = function_source[last_index + 1 :].lstrip()
            return function_source

        task_definition = CreateFromFunction.create_task_from_function(
            a_function=function,
            function_kwargs=function_kwargs or None,
            function_input_artifacts=function_input_artifacts,
            function_return=function_return,
            project_name=project_name,
            task_name=task_name,
            task_type=task_type,
            auto_connect_frameworks=auto_connect_frameworks,
            auto_connect_arg_parser=auto_connect_arg_parser,
            repo=repo,
            branch=branch,
            commit=commit,
            packages=packages,
            docker=docker,
            docker_args=docker_args,
            docker_bash_setup_script=docker_bash_setup_script,
            output_uri=output_uri,
            helper_functions=helper_functions,
            dry_run=True,
            task_template_header=self._task_template_header,
            _sanitize_function=sanitize,
            artifact_serialization_function=self._artifact_serialization_function,
            artifact_deserialization_function=self._artifact_deserialization_function,
            skip_global_imports=self._skip_global_imports,
            working_dir=working_dir,
        )
        return task_definition

    def _find_executed_node_leaves(self) -> List[PipelineController.Node]:
        all_parents = set([p for n in list(self._nodes.values()) if n.executed for p in n.parents])
        executed_leaves = [name for name, n in list(self._nodes.items()) if n.executed and name not in all_parents]
        return executed_leaves

    def _adjust_task_hashing(self, task_hash: dict) -> dict:
        """
        Fix the Task hashing so that parameters pointing to the current Task artifact are encoded using the
        hash content of the artifact, instead of the Task.id
        :param task_hash: Task representation dict
        :return: Adjusted Task representation dict
        """
        if task_hash.get("hyper_params"):
            updated_params = {}
            for k, v in task_hash["hyper_params"].items():
                if k.startswith("{}/".format(CreateFromFunction.input_artifact_section)) and str(v).startswith(
                    "{}.".format(self._task.id)
                ):
                    task_id, artifact_name = str(v).split(".", 1)
                    if artifact_name in self._task.artifacts:
                        updated_params[k] = self._task.artifacts[artifact_name].hash
            task_hash["hyper_params"].update(updated_params)

        return task_hash

    @classmethod
    def _wait_for_node(cls, node: PipelineController.Node) -> None:
        pool_period = 5.0 if cls._debug_execute_step_process else 20.0
        while True:
            if not node.job:
                break
            node.job.wait(pool_period=pool_period, aborted_nonresponsive_as_running=True)
            job_status = str(node.job.status(force=True))
            if (
                (
                    job_status == str(Task.TaskStatusEnum.stopped)
                    and node.job.status_message() == cls._relaunch_status_message
                )
                or (job_status == str(Task.TaskStatusEnum.failed) and not cls._final_failure.get(node.name))
                or not node.job.is_stopped()
            ):
                sleep(pool_period)
            else:
                break

    @classmethod
    def component(
        cls,
        _func: Any = None,
        *,
        return_values: Union[str, Sequence[str]] = ("return_object",),
        name: Optional[str] = None,
        cache: bool = False,
        packages: Optional[Union[bool, str, Sequence[str]]] = None,
        parents: Optional[List[str]] = None,
        execution_queue: Optional[str] = None,
        continue_on_fail: bool = False,
        docker: Optional[str] = None,
        docker_args: Optional[str] = None,
        docker_bash_setup_script: Optional[str] = None,
        task_type: Optional[str] = None,
        auto_connect_frameworks: Optional[dict] = None,
        auto_connect_arg_parser: Optional[dict] = None,
        repo: Optional[str] = None,
        repo_branch: Optional[str] = None,
        repo_commit: Optional[str] = None,
        helper_functions: Optional[Sequence[Callable]] = None,
        monitor_metrics: Optional[List[Union[Tuple[str, str], Tuple]]] = None,
        monitor_artifacts: Optional[List[Union[str, Tuple[str, str]]]] = None,
        monitor_models: Optional[List[Union[str, Tuple[str, str]]]] = None,
        retry_on_failure: Optional[
            Union[int, Callable[[PipelineController, PipelineController.Node, int], bool]]
        ] = None,  # noqa
        pre_execute_callback: Optional[
            Callable[[PipelineController, PipelineController.Node, dict], bool]
        ] = None,  # noqa
        post_execute_callback: Optional[Callable[[PipelineController, PipelineController.Node], None]] = None,
        # noqa
        status_change_callback: Optional[
            Callable[[PipelineController, PipelineController.Node, str], None]
        ] = None,  # noqa
        tags: Optional[Union[str, Sequence[str]]] = None,
        output_uri: Optional[Union[str, bool]] = None,
        draft: Optional[bool] = False,
        working_dir: Optional[str] = None,
        continue_behaviour: Optional[dict] = None,
        stage: Optional[str] = None
    ) -> Callable:
        """
        pipeline component function to be executed remotely

        :param _func: wrapper function
        :param return_values: Provide a list of names for all the results.
            Notice! If not provided, no results will be stored as artifacts.
        :param name: Optional, set the name of the pipeline component task.
            If not provided, the wrapped function name is used as the pipeline component name
        :param cache: If True, before launching the new step,
            after updating with the latest configuration, check if an exact Task with the same parameter/code
            was already executed. If it was found, use it instead of launching a new Task. Default: False
        :param packages: Manually specify a list of required packages or a local requirements.txt file.
            Example: ["tqdm>=2.1", "scikit-learn"] or "./requirements.txt"
            If not provided, packages are automatically added based on the imports used inside the wrapped function.
            Use `False` to install requirements from "requirements.txt" inside your git repository
        :param parents: Optional list of parent nodes in the DAG.
            The current step in the pipeline will be sent for execution only after all the parent nodes
            have been executed successfully.
        :param execution_queue: Optional, the queue to use for executing this specific step.
            If not provided, the task will be sent to the pipeline's default execution queue
        :param continue_on_fail: (Deprecated, use `continue_behaviour` instead).
            If True, failed step will not cause the pipeline to stop
            (or marked as failed). Notice, that steps that are connected (or indirectly connected)
            to the failed step will be skipped. Defaults to False
        :param docker: Specify the docker image to be used when executing the pipeline step remotely
        :param docker_args: Add docker execution arguments for the remote execution
            (use single string for all docker arguments).
        :param docker_bash_setup_script: Add a bash script to be executed inside the docker before
            setting up the Task's environment
        :param task_type: Optional, The task type to be created. Supported values: 'training', 'testing', 'inference',
            'data_processing', 'application', 'monitor', 'controller', 'optimizer', 'service', 'qc', 'custom'
        :param auto_connect_frameworks: Control the frameworks auto connect, see `Task.init` auto_connect_frameworks
        :param auto_connect_arg_parser: Control the ArgParser auto connect, see `Task.init` auto_connect_arg_parser
        :param repo: Optional, specify a repository to attach to the function, when remotely executing.
            Allow users to execute the function inside the specified repository, enabling them to load modules/script
            from the repository. Notice the execution work directory will be the repository root folder.
            Supports both git repo url link, and local repository path (automatically converted into the remote
            git/commit as is currently checkout).
            Example remote url: 'https://github.com/user/repo.git'
            Example local repo copy: './repo' -> will automatically store the remote
            repo url and commit ID based on the locally cloned copy
        :param repo_branch: Optional, specify the remote repository branch (Ignored, if local repo path is used)
        :param repo_commit: Optional, specify the repository commit ID (Ignored, if local repo path is used)
        :param helper_functions: Optional, a list of helper functions to make available
            for the standalone pipeline step function Task. By default the pipeline step function has
            no access to any of the other functions, by specifying additional functions here, the remote pipeline step
            could call the additional functions.
            Example, assuming we have two functions parse_data(), and load_data(): [parse_data, load_data]
        :param monitor_metrics: Optional, Automatically log the step's reported metrics also on the pipeline Task.
            The expected format is a list of pairs metric (title, series) to log: ``[(step_metric_title, step_metric_series), ]``.
            For example: ``[('test', 'accuracy'), ]``.
            Or a list of tuple pairs, to specify a different target metric to use on the pipeline Task:
            ``[((step_metric_title, step_metric_series), (target_metric_title, target_metric_series)), ]``.
            For example: ``[[('test', 'accuracy'), ('model', 'accuracy')], ]``
        :param monitor_artifacts: Optional, Automatically log the step's artifacts on the pipeline Task.
            Provided a list of artifact names created by the step function, these artifacts will be logged
            automatically also on the Pipeline Task itself.
            Example: ``['processed_data', ]``
            (target artifact name on the Pipeline Task will hav ethe same name as the original artifact).
            Alternatively, provide a list of pairs ``(source_artifact_name, target_artifact_name)``:
            where the first string is the artifact name as it appears on the component Task,
            and the second is the target artifact name to put on the Pipeline Task.
            Example: ``[('processed_data', 'final_processed_data'), ]``
        :param monitor_models: Optional, Automatically log the step's output models on the pipeline Task.
            Provided a list of model names created by the step's Task, they will also appear on the Pipeline itself.
            Example: ``['model_weights', ]``.
            To select the latest (lexicographic) model use "model_*", or the last created model with just "*".
            Example:  ``['model_weights_*', ]``.
            Alternatively, provide a list of pairs ``(source_model_name, target_model_name)``:
            where the first string is the model name as it appears on the component Task,
            and the second is the target model name to put on the Pipeline Task.
            Example: ``[('model_weights', 'final_model_weights'), ]``
        :param retry_on_failure: Integer (number of retries) or Callback function that returns True to allow a retry

          - Integer: In case of node failure, retry the node the number of times indicated by this parameter.
          - Callable: A function called on node failure. Takes as parameters:
            the PipelineController instance, the PipelineController.Node that failed and an int
            representing the number of previous retries for the node that failed
            The function must return a `bool`: True if the node should be retried and False otherwise.
            If True, the node will be re-queued and the number of retries left will be decremented by 1.
            By default, if this callback is not specified, the function will be retried the number of
            times indicated by `retry_on_failure`.

            .. code-block:: py

                def example_retry_on_failure_callback(pipeline, node, retries):
                    print(node.name, ' failed')
                    # allow up to 5 retries (total of 6 runs)
                    return retries < 5

        :param pre_execute_callback: Callback function, called when the step (Task) is created,
            and before it is sent for execution. Allows a user to modify the Task before launch.
            Use `node.job` to access the ClearmlJob object, or `node.job.task` to directly access the Task object.
            ``parameters`` are the configuration arguments passed to the ClearmlJob.

            If the callback returned value is `False`,
            the Node is skipped and so is any node in the DAG that relies on this node.

            Notice the `parameters` are already parsed,
            e.g. ``${step1.parameters.Args/param}`` is replaced with relevant value.

            .. code-block:: py

                def step_created_callback(
                    pipeline,             # type: PipelineController,
                    node,                 # type: PipelineController.Node,
                    parameters,           # type: dict
                ):
                    pass

        :param post_execute_callback: Callback function, called when a step (Task) is completed
            and other jobs are going to be executed. Allows a user to modify the Task status after completion.

            .. code-block:: py

                def step_completed_callback(
                    pipeline,             # type: PipelineController,
                    node,                 # type: PipelineController.Node,
                ):
                    pass

        :param status_change_callback: Callback function, called when the status of a step (Task) changes.
            Use `node.job` to access the ClearmlJob object, or `node.job.task` to directly access the Task object.
            The signature of the function must look the following way:

            .. code-block:: py

                def status_change_callback(
                    pipeline,             # type: PipelineController,
                    node,                 # type: PipelineController.Node,
                    previous_status       # type: str
                ):
                    pass

        :param tags: A list of tags for the specific pipeline step.
            When executing a Pipeline remotely
            (i.e. launching the pipeline from the UI/enqueuing it), this method has no effect.
        :param output_uri: The storage / output url for this step. This is the default location for output
            models and other artifacts. Check Task.init reference docs for more info (output_uri is a parameter).
        :param draft: (default False). If True, the Task will be created as a draft task.
        :param working_dir:  Working directory to launch the step from.
        :param continue_behaviour: Controls whether the pipeline will continue running after a step failed/was aborted.
            Different behaviours can be set using a dictionary of boolean options. Supported options are:

          - continue_on_fail - If True, the pipeline will continue even if the step failed.
            If False, the pipeline will stop
          - continue_on_abort - If True, the pipeline will continue even if the step was aborted.
            If False, the pipeline will stop
          - skip_children_on_fail - If True, the children of this step will be skipped if it failed.
            If False, the children will run even if this step failed.
            Any parameters passed from the failed step to its children will default to None
          - skip_children_on_abort - If True, the children of this step will be skipped if it was aborted.
            If False, the children will run even if this step was aborted.
           Any parameters passed from the failed step to its children will default to None
          - If the keys are not present in the dictionary, their values will default to True
        :param stage: Name of the stage. This parameter enables pipeline step grouping into stages

        :return: function wrapper
        """

        def decorator_wrap(func: Callable) -> Callable:
            if continue_on_fail:
                warnings.warn(
                    "`continue_on_fail` is deprecated. Use `continue_behaviour` instead",
                    DeprecationWarning,
                )

            # noinspection PyProtectedMember
            unwrapped_func = CreateFromFunction._deep_extract_wrapped(func)
            _name = name or str(unwrapped_func.__name__)
            function_return = return_values if isinstance(return_values, (tuple, list)) else [return_values]

            inspect_func = inspect.getfullargspec(unwrapped_func)
            # add default argument values
            if inspect_func.args:
                default_values = list(inspect_func.defaults or [])
                default_values = ([None] * (len(inspect_func.args) - len(default_values))) + default_values
                function_kwargs = {k: v for k, v in zip(inspect_func.args, default_values)}
            else:
                function_kwargs = dict()

            add_step_spec = dict(
                name=_name,
                function=func,
                function_kwargs=function_kwargs,
                function_return=function_return,
                cache_executed_step=cache,
                packages=packages,
                parents=parents,
                execution_queue=execution_queue,
                continue_on_fail=continue_on_fail,
                docker=docker,
                docker_args=docker_args,
                docker_bash_setup_script=docker_bash_setup_script,
                auto_connect_frameworks=auto_connect_frameworks,
                auto_connect_arg_parser=auto_connect_arg_parser,
                task_type=task_type,
                repo=repo,
                repo_branch=repo_branch,
                repo_commit=repo_commit,
                helper_functions=helper_functions,
                monitor_metrics=monitor_metrics,
                monitor_models=monitor_models,
                monitor_artifacts=monitor_artifacts,
                pre_execute_callback=pre_execute_callback,
                post_execute_callback=post_execute_callback,
                status_change_callback=status_change_callback,
                tags=tags,
                output_uri=output_uri,
                draft=draft,
                working_dir=working_dir,
                continue_behaviour=continue_behaviour,
                stage=stage
            )

            if cls._singleton:
                cls._singleton._add_function_step(**add_step_spec)
            else:
                cls._added_decorator.append(add_step_spec)

            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Union[LazyEvalWrapper, List[LazyEvalWrapper]]:
                if cls._debug_execute_step_function:
                    args = walk_nested_dict_tuple_list(
                        args,
                        lambda x: x._remoteref() if isinstance(x, LazyEvalWrapper) else x,
                    )
                    kwargs = walk_nested_dict_tuple_list(
                        kwargs,
                        lambda x: x._remoteref() if isinstance(x, LazyEvalWrapper) else x,
                    )

                    func_return = []

                    def result_wrapper(a_func_return: List[Any], return_index: Optional[int]) -> Any:
                        if not a_func_return:
                            a_func_return.append(func(*args, **kwargs))
                        a_func_return = a_func_return[0]
                        return a_func_return if return_index is None else a_func_return[return_index]

                    if len(function_return) == 1:
                        ret_val = LazyEvalWrapper(
                            callback=functools.partial(result_wrapper, func_return, None),
                            remote_reference=functools.partial(result_wrapper, func_return, None),
                        )
                        cls._ref_lazy_loader_id_to_node_name[id(ret_val)] = _name
                        return ret_val
                    else:
                        return_w = [
                            LazyEvalWrapper(
                                callback=functools.partial(result_wrapper, func_return, i),
                                remote_reference=functools.partial(result_wrapper, func_return, i),
                            )
                            for i, _ in enumerate(function_return)
                        ]
                        for i in return_w:
                            cls._ref_lazy_loader_id_to_node_name[id(i)] = _name
                        return return_w

                # resolve all lazy objects if we have any:
                kwargs_artifacts = {}
                star_args_index = 0
                for i, v in enumerate(args):
                    if not inspect_func.args or i >= len(inspect_func.args):
                        kwargs[str(star_args_index)] = v
                        star_args_index += 1
                    else:
                        kwargs[inspect_func.args[i]] = v

                # We need to remember when a pipeline step's return value is evaluated by the pipeline
                # controller, but not when it's done here (as we would remember the step every time).
                # _add_to_evaluated_return_values protects that
                tid = current_thread().ident
                cls._add_to_evaluated_return_values[tid] = False
                kwargs_artifacts.update(
                    {
                        k: walk_nested_dict_tuple_list(
                            v,
                            lambda x: x._remoteref() if isinstance(x, LazyEvalWrapper) else x,
                        )
                        for k, v in kwargs.items()
                        if isinstance(v, LazyEvalWrapper)
                    }
                )
                cls._add_to_evaluated_return_values[tid] = True
                kwargs = {k: deepcopy(v) for k, v in kwargs.items() if not isinstance(v, LazyEvalWrapper)}

                # check if we have the singleton
                if not cls._singleton:
                    # todo: somehow make sure the generated tasks list the parent pipeline as parent
                    original_tags = (
                        Task.current_task().get_tags(),
                        Task.current_task().get_system_tags(),
                    )
                    # This is an adhoc pipeline step,
                    PipelineDecorator._eager_execution_instance = True
                    a_pipeline = PipelineDecorator(
                        name=name,
                        project="DevOps",  # it will not actually be used
                        version="0.0.0",
                        pool_frequency=111,
                        add_pipeline_tags=False,
                        target_project=None,
                    )

                    target_queue = (
                        PipelineDecorator._default_execution_queue or Task.current_task().data.execution.queue
                    )
                    if target_queue:
                        PipelineDecorator.set_default_execution_queue(target_queue)
                    else:
                        # if we are not running from a queue, we are probably in debug mode
                        a_pipeline._clearml_job_class = LocalClearmlJob
                        a_pipeline._default_execution_queue = "mock"

                    # restore tags, the pipeline might add a few
                    Task.current_task().set_tags(original_tags[0])
                    Task.current_task().set_system_tags(original_tags[1])

                # get node name
                _node_name = _name

                # check if we are launching the same node twice
                if _node_name in cls._singleton._launched_step_names:
                    # if we already launched a JOB on the node, this means we are calling the same function/task
                    # twice inside the pipeline, this means we need to replicate the node.
                    _node = cls._singleton._nodes[_node_name].copy()
                    # reset paramters - there might be conflicts with the copied node and they are generated anyway
                    _node.parameters = {}
                    _node.parents = []
                    # find a new name
                    counter = 1
                    # Use nodes in `_singleton._nodes` that have not been launched.
                    # First check if we launched the node.
                    # If it wasn't launched we also need to check that the new name of `_node`
                    # points to the original code section it was meant to run.
                    # Note that for the first iteration (when `_node.name == _node_name`)
                    # we always increment the name, as the name is always in `_launched_step_names`
                    while _node.name in cls._singleton._launched_step_names or (
                        _node.name in cls._singleton._nodes
                        and cls._singleton._nodes[_node.name].job_code_section
                        != cls._singleton._nodes[_node_name].job_code_section
                    ):
                        _node.name = "{}_{}".format(_node_name, counter)
                        counter += 1
                    # Copy callbacks to the replicated node
                    if cls._singleton._pre_step_callbacks.get(_node_name):
                        cls._singleton._pre_step_callbacks[_node.name] = cls._singleton._pre_step_callbacks[_node_name]
                    if cls._singleton._post_step_callbacks.get(_node_name):
                        cls._singleton._post_step_callbacks[_node.name] = cls._singleton._post_step_callbacks[
                            _node_name
                        ]
                    if cls._singleton._status_change_callbacks.get(_node_name):
                        cls._singleton._status_change_callbacks[_node.name] = cls._singleton._status_change_callbacks[
                            _node_name
                        ]
                    _node_name = _node.name
                    if _node.name not in cls._singleton._nodes:
                        cls._singleton._nodes[_node.name] = _node

                # get node and park is as launched
                cls._singleton._launched_step_names.add(_node_name)
                _node = cls._singleton._nodes[_node_name]
                cls._retries[_node_name] = 0
                cls._retries_callbacks[_node_name] = (
                    retry_on_failure
                    if callable(retry_on_failure)
                    else (
                        functools.partial(
                            cls._singleton._default_retry_on_failure_callback,
                            max_retries=retry_on_failure,
                        )
                        if isinstance(retry_on_failure, int)
                        else cls._singleton._retry_on_failure_callback
                    )
                )

                # The actual launch is a bit slow, we run it in the background
                launch_thread = Thread(
                    target=cls._component_launch,
                    args=(
                        _node_name,
                        _node,
                        kwargs_artifacts,
                        kwargs,
                        current_thread().ident,
                    ),
                )

                def results_reference(return_name: str) -> str:
                    # wait until launch is completed
                    if launch_thread and launch_thread.is_alive():
                        try:
                            launch_thread.join()
                        except:  # noqa
                            pass

                    cls._wait_for_node(_node)
                    if not _node.job:
                        if not _node.executed:
                            raise ValueError("Job was not created and is also not cached/executed")
                        return "{}.{}".format(_node.executed, return_name)

                    if _node.job.is_failed() and not _node.continue_on_fail:
                        raise ValueError(
                            'Pipeline step "{}", Task ID={} failed'.format(_node.name, _node.job.task_id())
                        )

                    _node.executed = _node.job.task_id()
                    return "{}.{}".format(_node.job.task_id(), return_name)

                def result_wrapper(return_name: str) -> Any:
                    # wait until launch is completed
                    if launch_thread and launch_thread.is_alive():
                        try:
                            launch_thread.join()
                        except:  # noqa
                            pass

                    # skipped job
                    if not _node.job:
                        return None

                    cls._wait_for_node(_node)
                    if (_node.job.is_failed() and not _node.continue_on_fail) or (
                        _node.job.is_aborted() and not _node.continue_on_abort
                    ):
                        raise ValueError(
                            'Pipeline step "{}", Task ID={} failed'.format(_node.name, _node.job.task_id())
                        )

                    _node.executed = _node.job.task_id()

                    # make sure we mark the current state of the DAG execution tree
                    # so that later we can find the "parents" to the current node
                    _tid = current_thread().ident
                    if cls._add_to_evaluated_return_values.get(_tid, True):
                        if _tid not in cls._evaluated_return_values:
                            cls._evaluated_return_values[_tid] = []
                        cls._evaluated_return_values[_tid].append(_node.name)

                    task = Task.get_task(_node.job.task_id())
                    if return_name in task.artifacts:
                        return task.artifacts[return_name].get(
                            deserialization_function=cls._singleton._artifact_deserialization_function
                        )
                    return task.get_parameters(cast=True).get(CreateFromFunction.return_section + "/" + return_name)

                return_w = [
                    LazyEvalWrapper(
                        callback=functools.partial(result_wrapper, n),
                        remote_reference=functools.partial(results_reference, n),
                    )
                    for n in function_return
                ]
                for i in return_w:
                    cls._ref_lazy_loader_id_to_node_name[id(i)] = _node_name

                # start the launch thread now
                launch_thread.start()

                return return_w[0] if len(return_w) == 1 else return_w

            return wrapper

        return decorator_wrap if _func is None else decorator_wrap(_func)

    @classmethod
    def pipeline(
        cls,
        _func: Any = None,
        *,  # noqa
        name: str,
        project: str,
        version: Optional[str] = None,
        return_value: Optional[str] = None,
        default_queue: Optional[str] = None,
        pool_frequency: float = 0.2,
        add_pipeline_tags: bool = False,
        target_project: Optional[str] = None,
        abort_on_failure: bool = False,
        pipeline_execution_queue: Optional[str] = "services",
        multi_instance_support: bool = False,
        add_run_number: bool = True,
        args_map: Dict[str, List[str]] = None,
        start_controller_locally: bool = False,
        retry_on_failure: Optional[
            Union[int, Callable[[PipelineController, PipelineController.Node, int], bool]]
        ] = None,  # noqa
        docker: Optional[str] = None,
        docker_args: Optional[str] = None,
        docker_bash_setup_script: Optional[str] = None,
        packages: Optional[Union[bool, str, Sequence[str]]] = None,
        repo: Optional[str] = None,
        repo_branch: Optional[str] = None,
        repo_commit: Optional[str] = None,
        artifact_serialization_function: Optional[Callable[[Any], Union[bytes, bytearray]]] = None,
        artifact_deserialization_function: Optional[Callable[[bytes], Any]] = None,
        output_uri: Optional[Union[str, bool]] = None,
        skip_global_imports: bool = False,
        working_dir: Optional[str] = None,
        enable_local_imports: bool = True,
    ) -> Callable:
        """
        Decorate pipeline logic function.

        :param name: Provide pipeline name (if main Task exists it overrides its name)
        :param project: Provide project storing the pipeline (if main Task exists  it overrides its project)
        :param version: Pipeline version. This version allows to uniquely identify the pipeline
            template execution. Examples for semantic versions: version='1.0.1' , version='23', version='1.2'.
            If not set, find the latest version of the pipeline and increment it. If no such version is found,
            default to '1.0.0'
        :param return_value: Optional, Provide an artifact name to store the pipeline function return object
            Notice, If not provided the pipeline will not store the pipeline function return value.
        :param default_queue: default pipeline step queue
        :param float pool_frequency: The pooling frequency (in minutes) for monitoring experiments / states.
        :param bool add_pipeline_tags: (default: False) if True, add `pipe: <pipeline_task_id>` tag to all
            steps (Tasks) created by this pipeline.
        :param str target_project: If provided, all pipeline steps are cloned into the target project
        :param bool abort_on_failure: If False (default), failed pipeline steps will not cause the pipeline
            to stop immediately, instead any step that is not connected (or indirectly connected) to the failed step,
            will still be executed. Nonetheless, the pipeline itself will be marked failed, unless the failed step
            was specifically defined with "continue_on_fail=True".
            If True, any failed step will cause the pipeline to immediately abort, stop all running steps,
            and mark the pipeline as failed.
        :param pipeline_execution_queue: remote pipeline execution queue (default 'services' queue).
            If None is passed, execute the pipeline logic locally (pipeline steps are still executed remotely)
        :param multi_instance_support: If True, allow multiple calls to the same pipeline function,
            each call creating a new Pipeline Task. Notice it is recommended to create an additional Task on the
            "main process" acting as a master pipeline, automatically collecting the execution plots.
            If multi_instance_support=='parallel' then the pipeline calls are executed in parallel,
            in the `parallel` case the function calls return None, to collect all pipeline results call
            `PipelineDecorator.wait_for_multi_pipelines()`.
            Default False, no multi instance pipeline support.
        :param add_run_number: If True (default), add the run number of the pipeline to the pipeline name.
            Example, the second time we launch the pipeline "best pipeline", we rename it to "best pipeline #2"
        :param args_map: Map arguments to their specific configuration section. Arguments not included in this map
            will default to `Args` section. For example, for the following code:

            .. code-block:: py

                @PipelineDecorator.pipeline(args_map={'sectionA':['paramA'], 'sectionB:['paramB','paramC']
                def executing_pipeline(paramA, paramB, paramC, paramD):
                    pass

            Parameters would be stored as:

          - paramA: sectionA/paramA
          - paramB: sectionB/paramB
          - paramC: sectionB/paramC
          - paramD: Args/paramD

        :param start_controller_locally: If True, start the controller on the local machine. The steps will run
            remotely if `PipelineDecorator.run_locally` or `PipelineDecorator.debug_pipeline` are not called.
            Default: False
        :param retry_on_failure: Integer (number of retries) or Callback function that returns True to allow a retry

          - Integer: In case of node failure, retry the node the number of times indicated by this parameter.
          - Callable: A function called on node failure. Takes as parameters:
            the PipelineController instance, the PipelineController.Node that failed and an int
            representing the number of previous retries for the node that failed.
            The function must return ``True`` if the node should be retried and ``False`` otherwise.
            If True, the node will be re-queued and the number of retries left will be decremented by 1.
            By default, if this callback is not specified, the function will be retried the number of
            times indicated by `retry_on_failure`.

              .. code-block:: py

                  def example_retry_on_failure_callback(pipeline, node, retries):
                      print(node.name, ' failed')
                      # allow up to 5 retries (total of 6 runs)
                      return retries < 5
        :param docker: Select the docker image to be executed in by the remote session
        :param docker_args: Add docker arguments, pass a single string
        :param docker_bash_setup_script: Add bash script to be executed
            inside the docker before setting up the Task's environment
        :param packages: Manually specify a list of required packages or a local requirements.txt file.
            Example: ["tqdm>=2.1", "scikit-learn"] or "./requirements.txt"
            If not provided, packages are automatically added based on the imports used in the function.
            Use `False` to install requirements from "requirements.txt" inside your git repository
        :param repo: Optional, specify a repository to attach to the function, when remotely executing.
            Allow users to execute the function inside the specified repository, enabling them to load modules/script
            from the repository. Notice the execution work directory will be the repository root folder.
            Supports both git repo url link, and local repository path (automatically converted into the remote
            git/commit as is currently checkout).
            Example remote url: 'https://github.com/user/repo.git'
            Example local repo copy: './repo' -> will automatically store the remote
            repo url and commit ID based on the locally cloned copy
            Use empty string ("") to disable any repository auto-detection
        :param repo_branch: Optional, specify the remote repository branch (Ignored, if local repo path is used)
        :param repo_commit: Optional, specify the repository commit ID (Ignored, if local repo path is used)
        :param artifact_serialization_function: A serialization function that takes one
            parameter of any type which is the object to be serialized. The function should return
            a `bytes` or `bytearray` object, which represents the serialized object. All parameter/return
            artifacts uploaded by the pipeline will be serialized using this function.
            All relevant imports must be done in this function. For example:

            .. code-block:: py

                def serialize(obj):
                    import dill
                    return dill.dumps(obj)
        :param artifact_deserialization_function: A deserialization function that takes one parameter of type `bytes`,
            which represents the serialized object. This function should return the deserialized object.
            All parameter/return artifacts fetched by the pipeline will be deserialized using this function.
            All relevant imports must be done in this function. For example:

            .. code-block:: py

                def deserialize(bytes_):
                    import dill
                    return dill.loads(bytes_)
        :param output_uri: The storage / output url for this pipeline. This is the default location for output
            models and other artifacts. Check Task.init reference docs for more info (output_uri is a parameter).
            The `output_uri` of this pipeline's steps will default to this value.
        :param skip_global_imports: If True, global imports will not be included in the steps' execution, otherwise all
            global imports will be automatically imported in a safe manner at the beginning of each step’s execution.
            Default is False
        :param working_dir:  Working directory to launch the pipeline from.
        :param enable_local_imports: If True, allow pipeline steps to import from local files
            by appending to the PYTHONPATH of each step the directory the pipeline controller
            script resides in (sys.path[0]).
            If False, the directory won't be appended to PYTHONPATH. Default is True.
            Ignored while running remotely.
        """

        def decorator_wrap(func: Callable) -> Callable:
            def internal_decorator(*args: Any, **kwargs: Any) -> Any:
                pipeline_kwargs = dict(**(kwargs or {}))
                pipeline_kwargs_types = dict()
                inspect_func = inspect.getfullargspec(func)
                if args:
                    if not inspect_func.args:
                        raise ValueError("Could not parse function arguments")

                    pipeline_kwargs.update({inspect_func.args[i]: v for i, v in enumerate(args)})

                # add default function arguments if we have defaults for all arguments
                if inspect_func.args:
                    default_values = list(inspect_func.defaults or [])
                    default_values = ([None] * (len(inspect_func.args) - len(default_values))) + default_values
                    default_kwargs = {k: v for k, v in zip(inspect_func.args, default_values)}
                    default_kwargs.update(pipeline_kwargs)
                    pipeline_kwargs = default_kwargs

                if inspect_func.annotations:
                    pipeline_kwargs_types = {str(k): inspect_func.annotations[k] for k in inspect_func.annotations}

                # run the entire pipeline locally, as python functions
                if cls._debug_execute_step_function:
                    a_pipeline = PipelineDecorator(
                        name=name,
                        project=project,
                        version=version,
                        pool_frequency=pool_frequency,
                        add_pipeline_tags=add_pipeline_tags,
                        target_project=target_project,
                        abort_on_failure=abort_on_failure,
                        add_run_number=add_run_number,
                        retry_on_failure=retry_on_failure,
                        docker=docker,
                        docker_args=docker_args,
                        docker_bash_setup_script=docker_bash_setup_script,
                        packages=packages,
                        repo=repo,
                        repo_branch=repo_branch,
                        repo_commit=repo_commit,
                        artifact_serialization_function=artifact_serialization_function,
                        artifact_deserialization_function=artifact_deserialization_function,
                        output_uri=output_uri,
                        skip_global_imports=skip_global_imports,
                        working_dir=working_dir,
                        enable_local_imports=enable_local_imports,
                    )
                    ret_val = func(**pipeline_kwargs)
                    LazyEvalWrapper.trigger_all_remote_references()
                    a_pipeline._task.close()
                    return ret_val

                # check if we are in a multi pipeline
                force_single_multi_pipeline_call = False
                if multi_instance_support and cls._multi_pipeline_call_counter >= 0:
                    # check if we are running remotely
                    if not Task.running_locally():
                        # get the main Task property
                        t = Task.get_task(task_id=get_remote_task_id())
                        if str(t.task_type) == str(Task.TaskTypes.controller):
                            # noinspection PyBroadException
                            try:
                                # noinspection PyProtectedMember
                                multi_pipeline_call_counter = int(
                                    t._get_runtime_properties().get("multi_pipeline_counter", None)
                                )

                                # NOTICE! if this is not our call we LEAVE immediately
                                # check if this is our call to start, if not we will wait for the next one
                                if multi_pipeline_call_counter != cls._multi_pipeline_call_counter:
                                    return
                            except Exception:
                                # this is not the one, so we should just run the first
                                # instance and leave immediately
                                force_single_multi_pipeline_call = True

                if default_queue:
                    cls.set_default_execution_queue(default_queue)

                a_pipeline = PipelineDecorator(
                    name=name,
                    project=project,
                    version=version,
                    pool_frequency=pool_frequency,
                    add_pipeline_tags=add_pipeline_tags,
                    target_project=target_project,
                    abort_on_failure=abort_on_failure,
                    add_run_number=add_run_number,
                    retry_on_failure=retry_on_failure,
                    docker=docker,
                    docker_args=docker_args,
                    docker_bash_setup_script=docker_bash_setup_script,
                    packages=packages,
                    repo=repo,
                    repo_branch=repo_branch,
                    repo_commit=repo_commit,
                    artifact_serialization_function=artifact_serialization_function,
                    artifact_deserialization_function=artifact_deserialization_function,
                    output_uri=output_uri,
                    skip_global_imports=skip_global_imports,
                    working_dir=working_dir,
                    enable_local_imports=enable_local_imports,
                )

                a_pipeline._args_map = args_map or {}

                if PipelineDecorator._debug_execute_step_process:
                    a_pipeline._clearml_job_class = LocalClearmlJob
                    a_pipeline._default_execution_queue = "mock"

                a_pipeline._clearml_job_class.register_hashing_callback(a_pipeline._adjust_task_hashing)

                # add pipeline arguments
                for k in pipeline_kwargs:
                    a_pipeline.add_parameter(
                        name=k,
                        default=pipeline_kwargs.get(k),
                        param_type=pipeline_kwargs_types.get(k),
                    )

                # sync multi-pipeline call counter (so we know which one to skip)
                if Task.running_locally() and multi_instance_support and cls._multi_pipeline_call_counter >= 0:
                    # noinspection PyProtectedMember
                    a_pipeline._task._set_runtime_properties(
                        dict(multi_pipeline_counter=str(cls._multi_pipeline_call_counter))
                    )

                # run the actual pipeline
                if (
                    not start_controller_locally
                    and not PipelineDecorator._debug_execute_step_process
                    and pipeline_execution_queue
                ):
                    # rerun the pipeline on a remote machine
                    a_pipeline._task.execute_remotely(queue_name=pipeline_execution_queue)
                    # when we get here it means we are running remotely

                # this will also deserialize the pipeline and arguments
                a_pipeline._start(wait=False)

                # sync arguments back (post deserialization and casting back)
                for k in pipeline_kwargs.keys():
                    if k in a_pipeline.get_parameters():
                        pipeline_kwargs[k] = a_pipeline.get_parameters()[k]

                # this time the pipeline is executed only on the remote machine
                try:
                    pipeline_result = func(**pipeline_kwargs)
                except Exception:
                    a_pipeline.stop(mark_failed=True)
                    raise

                triggered_exception = None
                try:
                    LazyEvalWrapper.trigger_all_remote_references()
                except Exception as ex:
                    triggered_exception = ex

                # make sure we wait for all nodes to finish
                waited = True
                while waited:
                    waited = False
                    for node in list(a_pipeline._nodes.values()):
                        if node.executed or not node.job or node.job.is_stopped(aborted_nonresponsive_as_running=True):
                            continue
                        cls._wait_for_node(node)
                        waited = True
                # store the pipeline result of we have any:
                if return_value and pipeline_result is not None:
                    a_pipeline._upload_pipeline_artifact(
                        artifact_name=str(return_value), artifact_object=pipeline_result
                    )

                # now we can stop the pipeline
                a_pipeline.stop()
                # now we can raise the exception
                if triggered_exception:
                    raise triggered_exception

                # Make sure that if we do not need to run all pipelines we forcefully leave the process
                if force_single_multi_pipeline_call:
                    leave_process()
                    # we will never get here

                return pipeline_result

            if multi_instance_support:
                return cls._multi_pipeline_wrapper(
                    func=internal_decorator,
                    parallel=bool(multi_instance_support == "parallel"),
                )

            return internal_decorator

        return decorator_wrap if _func is None else decorator_wrap(_func)

    @classmethod
    def set_default_execution_queue(cls, default_execution_queue: Optional[str]) -> None:
        """
        Set the default execution queue if pipeline step does not specify an execution queue

        :param default_execution_queue: The execution queue to use if no execution queue is provided
        """
        cls._default_execution_queue = str(default_execution_queue) if default_execution_queue else None

    @classmethod
    def run_locally(cls) -> ():
        """
        Set local mode, run all functions locally as subprocess

        Run the full pipeline DAG locally, where steps are executed as sub-processes Tasks
        Notice: running the DAG locally assumes the local code execution (i.e. it will not clone & apply git diff)

        """
        cls._debug_execute_step_process = True
        cls._debug_execute_step_function = False

    @classmethod
    def debug_pipeline(cls) -> ():
        """
        Set debugging mode, run all functions locally as functions (serially)
        Run the full pipeline DAG locally, where steps are executed as functions

        .. note::
            Running the DAG locally assumes local code execution (i.e. it will not clone & apply git diff).
            Pipeline steps are executed as functions (no Task will be created).
        """
        cls._debug_execute_step_process = True
        cls._debug_execute_step_function = True

    @classmethod
    def get_current_pipeline(cls) -> "PipelineDecorator":
        """
        Return the currently running pipeline instance
        """
        return cls._singleton

    @classmethod
    def wait_for_multi_pipelines(cls) -> List[Any]:
        # type () -> List[object]
        """
        Wait until all background multi pipeline execution is completed.
        Returns all the pipeline results in call order (first pipeline call at index 0)

        :return: List of return values from executed pipeline, based on call order.
        """
        return cls._wait_for_multi_pipelines()

    @classmethod
    def _component_launch(
        cls,
        node_name: str,
        node: PipelineController.Node,
        kwargs_artifacts: Dict[str, Any],
        kwargs: Dict[str, Any],
        tid: int,
    ) -> None:
        _node_name = node_name
        _node = node
        # update artifacts kwargs
        for k, v in kwargs_artifacts.items():
            if k in kwargs:
                kwargs.pop(k, None)
            _node.parameters.pop("{}/{}".format(CreateFromFunction.kwargs_section, k), None)
            _node.parameters["{}/{}".format(CreateFromFunction.input_artifact_section, k)] = v
            if v and "." in str(v):
                parent_id, _ = str(v).split(".", 1)
                # find parent and push it into the _node.parents
                for n, node in sorted(list(cls._singleton._nodes.items()), reverse=True):
                    if n != _node.name and node.executed and node.executed == parent_id:
                        if n not in _node.parents:
                            _node.parents.append(n)
                        break

        leaves = cls._singleton._find_executed_node_leaves()
        _node.parents = (_node.parents or []) + [x for x in cls._evaluated_return_values.get(tid, []) if x in leaves]

        if not cls._singleton._abort_running_steps_on_failure:
            for parent in _node.parents:
                parent = cls._singleton._nodes[parent]
                if (
                    parent.status == "failed"
                    and parent.skip_children_on_fail
                    or parent.status == "aborted"
                    and parent.skip_children_on_abort
                    or parent.status == "skipped"
                ):
                    _node.skip_job = True
                    return

        for k, v in kwargs.items():
            if v is None or isinstance(v, (float, int, bool, six.string_types)):
                _node.parameters["{}/{}".format(CreateFromFunction.kwargs_section, k)] = v
            else:
                # we need to create an artifact
                artifact_name = "result_{}_{}".format(re.sub(r"\W+", "", _node.name), k)
                cls._singleton._upload_pipeline_artifact(artifact_name=artifact_name, artifact_object=v)
                _node.parameters["{}/{}".format(CreateFromFunction.input_artifact_section, k)] = "{}.{}".format(
                    cls._singleton._task.id, artifact_name
                )

        # verify the new step
        cls._singleton._verify_node(_node)
        # launch the new step
        cls._singleton._launch_node(_node)
        # check if we generated the pipeline we need to update the new eager step
        if PipelineDecorator._eager_execution_instance and _node.job:
            # check if we need to add the pipeline tag on the new node
            pipeline_tags = [t for t in Task.current_task().get_tags() or [] if str(t).startswith(cls._node_tag_prefix)]
            if pipeline_tags and _node.job and _node.job.task:
                pipeline_tags = list(set((_node.job.task.get_tags() or []) + pipeline_tags))
                _node.job.task.set_tags(pipeline_tags)
            # force parent task as pipeline
            _node.job.task._edit(parent=Task.current_task().parent)
            # store the new generated node, so we can later serialize it
            pipeline_dag = cls._singleton._serialize()
            # check if node is cached
            if _node.job.is_cached_task():
                pipeline_dag[_node_name]["is_cached"] = True
            # store entire definition on the parent pipeline
            from clearml.backend_api.services import tasks

            artifact = tasks.Artifact(
                key="{}:{}:{}".format(
                    cls._eager_step_artifact,
                    Task.current_task().id,
                    _node.job.task_id(),
                ),
                type="json",
                mode="output",
                type_data=tasks.ArtifactTypeData(
                    preview=json.dumps({_node_name: pipeline_dag[_node_name]}),
                    content_type="application/pipeline",
                ),
            )
            req = tasks.AddOrUpdateArtifactsRequest(task=Task.current_task().parent, artifacts=[artifact], force=True)
            res = Task.current_task().send(req, raise_on_errors=False)
            if not res or not res.response or not res.response.updated:
                pass

        # update pipeline execution graph
        cls._singleton.update_execution_plot()

    @classmethod
    def _multi_pipeline_wrapper(
        cls,
        func: Callable = None,
        parallel: bool = False,
    ) -> Callable:
        """
        Add support for multiple pipeline function calls,
        enabling execute multiple instances of the same pipeline from a single script.

        .. code-block:: py

            @PipelineDecorator.pipeline(
                multi_instance_support=True, name="custom pipeline logic", project="examples", version="1.0")
            def pipeline(parameter=1):
                print(f"running with parameter={parameter}")

            # run both pipeline (if multi_instance_support=='parallel', run pipelines in parallel)
            pipeline(parameter=1)
            pipeline(parameter=2)

        :param parallel: If True, the pipeline is running in the background, which implies calling
            the pipeline twice means running the pipelines in parallel.
            Default: False, pipeline function returns when pipeline completes
        :return: Return wrapped pipeline function.
            Notice the return value of the pipeline wrapped function:
            if parallel==True, return will be None, otherwise expect the return of the pipeline wrapped function
        """

        def internal_decorator(*args: Any, **kwargs: Any) -> Any:
            cls._multi_pipeline_call_counter += 1

            # if this is a debug run just call the function (no parallelization).
            if cls._debug_execute_step_function:
                return func(*args, **kwargs)

            def sanitized_env(a_queue: Queue, *a_args: Any, **a_kwargs: Any) -> Any:
                os.environ.pop("CLEARML_PROC_MASTER_ID", None)
                os.environ.pop("TRAINS_PROC_MASTER_ID", None)
                os.environ.pop("CLEARML_TASK_ID", None)
                os.environ.pop("TRAINS_TASK_ID", None)
                if Task.current_task():
                    # noinspection PyProtectedMember
                    Task.current_task()._reset_current_task_obj()
                a_result = func(*a_args, **a_kwargs)
                if a_queue is not None:
                    task_id = Task.current_task().id if Task.current_task() else None
                    a_queue.put((task_id, a_result))
                return a_result

            queue = Queue()

            p = Process(target=sanitized_env, args=(queue,) + args, kwargs=kwargs)
            # make sure we wait for the subprocess.
            p.daemon = False
            p.start()
            if parallel and Task.running_locally():
                cls._multi_pipeline_instances.append((p, queue))
                return
            else:
                p.join()
                # noinspection PyBroadException
                try:
                    pipeline_task, result = queue.get_nowait()
                except Exception:
                    return None

                # we should update the master Task plot:
                if pipeline_task and Task.current_task():
                    cls._add_pipeline_plots(pipeline_task)

                return result

        if parallel and not cls._atexit_registered:
            cls._atexit_registered = True
            atexit.register(cls._wait_for_multi_pipelines)

        return internal_decorator

    @classmethod
    def _wait_for_multi_pipelines(cls) -> List[Any]:
        results = []
        if not cls._multi_pipeline_instances:
            return results
        print("Waiting for background pipelines to finish")
        for p, queue in cls._multi_pipeline_instances:
            try:
                p.join()
            except:  # noqa
                pass
            # noinspection PyBroadException
            try:
                pipeline_task, result = queue.get_nowait()
                results.append(result)
                cls._add_pipeline_plots(pipeline_task)
            except Exception:
                pass
        cls._multi_pipeline_instances = []
        return results

    @classmethod
    def _add_pipeline_plots(cls, pipeline_task_id: str) -> None:
        if not Task.current_task():
            return
        from clearml.backend_api.services import events

        res = Task.current_task().send(
            events.GetTaskPlotsRequest(task=pipeline_task_id, iters=1),
            raise_on_errors=False,
            ignore_errors=True,
        )
        execution_flow = None
        execution_details = None
        for p in res.response.plots:
            try:
                if (
                    p["metric"] == cls._report_plot_execution_flow["title"]
                    and p["variant"] == cls._report_plot_execution_flow["series"]
                ):
                    execution_flow = json.loads(p["plot_str"])

                elif (
                    p["metric"] == cls._report_plot_execution_details["title"]
                    and p["variant"] == cls._report_plot_execution_details["series"]
                ):
                    execution_details = json.loads(p["plot_str"])
                    execution_details["layout"]["name"] += " - " + str(pipeline_task_id)
            except Exception as ex:
                getLogger("clearml.automation.controller").warning("Multi-pipeline plot update failed: {}".format(ex))

        if execution_flow:
            Task.current_task().get_logger().report_plotly(
                title=cls._report_plot_execution_flow["title"],
                series="{} - {}".format(cls._report_plot_execution_flow["series"], pipeline_task_id),
                iteration=0,
                figure=execution_flow,
            )
        if execution_details:
            Task.current_task().get_logger().report_plotly(
                title=cls._report_plot_execution_details["title"],
                series="{} - {}".format(cls._report_plot_execution_details["series"], pipeline_task_id),
                iteration=0,
                figure=execution_details,
            )
