"""Define tasks to be used by the library pydoit."""
from pydoit_project_builder import TaskCreator

# Define project parameters.
project_name = "nn-trainer"
python_version = "3"
# Instantiate tasks for pydoit library.
task_creator = TaskCreator(
    project_name=project_name, python_version=python_version)
list_callbacks = task_creator.get_all_tasks()
for callback in list_callbacks:
    fn_name = callback.__name__
    globals()["task_" + fn_name] = callback
