import json
import sys
import os
import pandas as pd
import importlib.util
import textwrap
from pathlib import Path
import inspect
from typing import get_type_hints, get_args, Annotated, Dict, Any
import re

REGISTRY_PATH = "./pipeline_zoo.json"
PIPELINE_PATH = "./extracted_pipelines.py"

def import_pipeline_function(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    # sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def extract_registry_info(func):
    signature = inspect.signature(func)
    docstring = inspect.getdoc(func)

    args_info = {}
    hints = get_type_hints(func, include_extras=True)
    for param_name, param in signature.parameters.items():
        annotation = hints.get(param_name, str)
        
        # Check for Annotated type to extract description
        if hasattr(annotation, '__origin__') and annotation.__origin__ is Annotated:
            arg_type, arg_description = get_args(annotation)
        else:
            arg_type, arg_description = annotation, ""

        default = param.default if param.default != inspect.Parameter.empty else "Required"
        args_info[param_name] = {
            "type": arg_type.__name__,
            "description": arg_description,
            "default": default if default != "Required" else None
        }

    registry = {
        "function_name": func.__name__,
        "description": docstring,
        "args": args_info,
        "output": "Dict[str, Any]"  # You can parse more specifically if needed
    }
    return registry

def get_log(dbname="logs.db", table="chat_completions"):
    import sqlite3

    con = sqlite3.connect(dbname)
    query = f"SELECT * from {table}"
    cursor = con.execute(query)
    rows = cursor.fetchall()
    column_names = [description[0] for description in cursor.description]
    data = [dict(zip(column_names, row)) for row in rows]
    con.close()
    return data

def str_to_dict(s):
    return json.loads(s)

def load_chat_log(chat_log_path):
    if os.path.splitext(os.path.basename(chat_log_path))[-1] == '.log':
        log_data = []
        # Read the file line by line and parse JSON entries
        with open(chat_log_path, "r") as file:
            for line in file:
                try:
                    log_entry = json.loads(line.strip())
                    log_data.append(log_entry)
                except json.JSONDecodeError:
                    continue  # Skip lines that aren't valid JSON
        return log_data
    elif os.path.splitext(os.path.basename(chat_log_path))[-1] == '.db': # db
        log_data = get_log(dbname=chat_log_path)
        log_data_df = pd.DataFrame(log_data)
        log_data_df["total_tokens"] = log_data_df.apply(
            lambda row: str_to_dict(row["response"])["usage"]["total_tokens"], axis=1
        )
        log_data_df["request"] = log_data_df.apply(lambda row: str_to_dict(row["request"])["messages"][0]["content"], axis=1)
        log_data_df["response"] = log_data_df.apply(
            lambda row: str_to_dict(row["response"])["choices"][0]["message"]["content"], axis=1
        )
        return log_data_df
    else:
        raise ValueError("Invalid chat log file format. Supported formats are .log and .db.")

def save_pipeline(pipeline_name, 
                  pipeline_code,
                  ):
    # os.makedirs(PIPELINE_DIR, exist_ok=True)
    # file_path = os.path.join(PIPELINE_DIR, f"{pipeline_name}.py")
    file_path = PIPELINE_PATH

    # Load existing registry if pipeline exists
    if os.path.exists(REGISTRY_PATH):
        with open(REGISTRY_PATH, 'r') as f:
            registry = json.load(f)
    else:
        registry = {}
    if pipeline_name in registry:
        raise ValueError(f"Pipeline '{pipeline_name}' already exists in the registry. Choose a different name or delete the existing one.") # or return a message

    # Clean code indentation
    pipeline_code = textwrap.dedent(pipeline_code)
    if pipeline_code.startswith("```") and pipeline_code.endswith("```"):
        pipeline_code = re.sub(r"^```(python)?\n?", "", pipeline_code.strip())  # Remove leading triple backticks and optional "python"
        pipeline_code = re.sub(r"\n```$", "", pipeline_code.strip()) # Remove trailing triple backticks

    # Save pipeline function to file
    with open(file_path, 'a') as f:
        f.write("\n\n" + pipeline_code.strip() + "\n")

    # inspect the pipeline code to extract function name, description, args, and output
    module = import_pipeline_function(
        pipeline_name,
        file_path,
    )
    pipeline_func = getattr(module, pipeline_name)
    registry_info = extract_registry_info(pipeline_func)
    
    # update registry
    registry[pipeline_name] = registry_info
    with open(REGISTRY_PATH, 'w') as f:
        json.dump(registry, f, indent=4)

    print(f"Pipeline '{pipeline_name}' saved and registered successfully.")

def get_pipeline_zoo() -> Dict[str, Any]:
    '''Return the information of all regsitered pipelines.'''
    with open(REGISTRY_PATH, 'r') as f:
        registry = json.load(f)
    return registry

def get_pipeline_info(pipeline_name: Annotated[str, "Name of pipeline to be queried."]) -> Dict[str, Any]:
    '''Return the information of the queried pipeline.'''
    with open(REGISTRY_PATH, 'r') as f:
        registry = json.load(f)
    pipeline_info = registry.get(pipeline_name)
    if pipeline_info is None:
        raise ValueError(f"Pipeline '{pipeline_name}' not found.")
    return pipeline_info

def execute_pipeline(pipeline_name: Annotated[str, "Name of pipeline to be called."],
                     **kwargs: Annotated[Dict[str, Any], "Arguments of the selected pipeline."]) -> Any:
    '''Run a pipeline with customised arguments.'''    
    try:
        with open(REGISTRY_PATH, 'r') as f:
            registry = json.load(f)
        pipeline_info = registry.get(pipeline_name)
        if pipeline_info is None:
            raise ValueError(f"Pipeline '{pipeline_name}' not found in registry.")
        
        spec = importlib.util.spec_from_file_location(
            pipeline_info['function_name'], PIPELINE_PATH
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        pipeline_func = getattr(module, pipeline_info['function_name'])

        # Call pipeline function
        result = pipeline_func(**kwargs)
        return result
    except Exception as e:
        return str(e)  # Indicate failure with error message