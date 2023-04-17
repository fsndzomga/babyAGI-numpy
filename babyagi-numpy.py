#!/usr/bin/env python3
import os
import subprocess
import time
from collections import deque
from typing import Dict, List
import importlib
import numpy as np


import openai
from dotenv import load_dotenv

# Load default environment variables (.env)
load_dotenv()

# Engine configuration

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
assert OPENAI_API_KEY, "OPENAI_API_KEY environment variable is missing from .env"

OPENAI_API_MODEL = os.getenv("OPENAI_API_MODEL", "gpt-3.5-turbo")
assert OPENAI_API_MODEL, "OPENAI_API_MODEL environment variable is missing from .env"

if "gpt-4" in OPENAI_API_MODEL.lower():
    print(
        "\033[91m\033[1m"
        + "\n*****USING GPT-4. POTENTIALLY EXPENSIVE. MONITOR YOUR COSTS*****"
        + "\033[0m\033[0m"
    )


# Goal configuation
OBJECTIVE = os.getenv("OBJECTIVE", "")
INITIAL_TASK = os.getenv("INITIAL_TASK", os.getenv("FIRST_TASK", ""))

# Model configuration
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", 0.0))


# Extensions support begin

def can_import(module_name):
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False


# Command line arguments extension
# Can override any of the above environment variables
ENABLE_COMMAND_LINE_ARGS = (
    os.getenv("ENABLE_COMMAND_LINE_ARGS", "false").lower() == "true"
)


# Check if we know what we are doing
assert OBJECTIVE, "OBJECTIVE environment variable is missing from .env"
assert INITIAL_TASK, "INITIAL_TASK environment variable is missing from .env"

if "gpt-4" in OPENAI_API_MODEL.lower():
    print(
        "\033[91m\033[1m"
        + "\n*****USING GPT-4. POTENTIALLY EXPENSIVE. MONITOR YOUR COSTS*****"
        + "\033[0m\033[0m"
    )

# Print OBJECTIVE
print("\033[94m\033[1m" + "\n*****OBJECTIVE*****\n" + "\033[0m\033[0m")
print(f"{OBJECTIVE}")

print("\033[93m\033[1m" + "\nInitial task:" + "\033[0m\033[0m" + f" {INITIAL_TASK}")

# Configure OpenAI
openai.api_key = OPENAI_API_KEY


results_store = {}

# Task list
task_list = deque([])


def add_task(task: Dict):
    task_list.append(task)


def get_ada_embedding(text):
    text = text.replace("\n", " ")
    return openai.Embedding.create(input=[text], model="text-embedding-ada-002")[
        "data"
    ][0]["embedding"]


def openai_call(
    prompt: str,
    model: str = OPENAI_API_MODEL,
    temperature: float = OPENAI_TEMPERATURE,
    max_tokens: int = 100,
):
    while True:
        try:
            if model.startswith("llama"):
                # Spawn a subprocess to run llama.cpp
                cmd = ["llama/main", "-p", prompt]
                result = subprocess.run(cmd, shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.PIPE, text=True)
                return result.stdout.strip()
            elif not model.startswith("gpt-"):
                # Use completion API
                response = openai.Completion.create(
                    engine=model,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                )
                return response.choices[0].text.strip()
            else:
                # Use chat completion API
                messages = [{"role": "system", "content": prompt}]
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    n=1,
                    stop=None,
                )
                return response.choices[0].message.content.strip()
        except openai.error.RateLimitError:
            print(
                "The OpenAI API rate limit has been exceeded. Waiting 10 seconds and trying again."
            )
            time.sleep(10)  # Wait 10 seconds and try again
        else:
            break

def define_success_criteria(objective: str) -> Dict[str, Dict[str, bool]]:
    prompt = f"Based on the objective '{objective}', define success criteria as a list:"
    success_criteria = openai_call(prompt)
    criteria_list = success_criteria.split("\n") if "\n" in success_criteria else [success_criteria]

    # Create a dictionary with success criteria and their validation statuses
    criteria_dict = {}
    for criterion in criteria_list:
        criteria_dict[criterion] = {'validated': False}

    return criteria_dict


def evaluation_agent(objective: str, results_store: Dict, success_criteria: Dict):
    # Get the latest result
    last_result_id = max(results_store, key=lambda x: int(x.split('_')[1]))
    last_result = results_store[last_result_id]['task']

    # Initialize met_count with the number of success criteria with 'validated' value being True
    met_count = sum([1 for value in success_criteria.values() if value['validated']])

    total_criteria = len(success_criteria)
    for criterion, value in success_criteria.items():
        if not value['validated']:
            prompt = f"Based on the latest result:\n{last_result}\nCheck if the success criterion '{criterion}' is met. Respond with just one word yes or no:"
            is_met = openai_call(prompt)
            if is_met.lower() == "yes":
                success_criteria[criterion]['validated'] = True
                met_count += 1

    # Check if more or equal to 80% of success criteria have been met
    return met_count >= 0.8 * total_criteria



def task_creation_agent(
    objective: str, result: Dict, task_description: str, task_list: List[str], success_criteria: Dict
):
    # Filter only the incomplete success criteria
    incomplete_criteria = [k for k, v in success_criteria.items() if not v["validated"]]

    prompt = f"""
    You are a task creation AI that uses the result of an execution agent to create new tasks with the following objective: {objective}.
    The last completed task has the result: {result}.
    This result was based on this task description: {task_description}. These are incomplete tasks: {', '.join(task_list)}.
    Imperatively take into account the fact that these success criteria are not yet met: {', '.join(incomplete_criteria)}.
    Based on the result and incomplete success criteria, create new tasks to be completed by the AI system that do not overlap with incomplete tasks.
    Return the tasks as an array."""
    response = openai_call(prompt)
    new_tasks = response.split("\n") if "\n" in response else [response]
    return [{"task_name": task_name} for task_name in new_tasks]



def prioritization_agent(this_task_id: int):
    global task_list
    task_names = [t["task_name"] for t in task_list]
    next_task_id = int(this_task_id) + 1
    # Filter only the incomplete success criteria
    incomplete_criteria = [k for k, v in success_criteria.items() if not v["validated"]]
    prompt = f"""
    You are a task prioritization AI tasked with cleaning the formatting of and reprioritizing the following tasks: {task_names}.
    Consider the ultimate objective of your team:{OBJECTIVE}.
    Do not remove any tasks. Return the result as a numbered list, like:
    #. First task
    #. Second task
    Start the task list with number {next_task_id}."""
    response = openai_call(prompt)
    new_tasks = response.split("\n") if "\n" in response else [response]
    task_list = deque()
    for task_string in new_tasks:
        task_parts = task_string.strip().split(".", 1)
        if len(task_parts) == 2:
            task_id = task_parts[0].strip()
            task_name = task_parts[1].strip()
            task_list.append({"task_id": task_id, "task_name": task_name})


def execution_agent(objective: str, task: str) -> str:
    """
    Executes a task based on the given objective and previous context.
    Args:
        objective (str): The objective or goal for the AI to perform the task.
        task (str): The task to be executed by the AI.
    Returns:
        str: The response generated by the AI for the given task.
    """

    context = context_agent(query=objective, top_results_num=5)
    # print("\n*******RELEVANT CONTEXT******\n")
    # print(context)
    prompt = f"""
    You are an AI who performs one task based on the following objective: {objective}\n.
    Take into account these previously completed tasks: {context}\n.
    Your task: {task}\nResponse:"""
    return openai_call(prompt, max_tokens=2000)


def context_agent(query: str, top_results_num: int):
    query_embedding = get_ada_embedding(query)
    similarities = {}
    for result_id, value in results_store.items():
        similarity = np.dot(query_embedding, value['vector']) / (np.linalg.norm(query_embedding) * np.linalg.norm(value['vector']))
        similarities[result_id] = similarity
    sorted_results = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_results_num]
    return [(results_store[item[0]]['task']) for item in sorted_results]


start_time = time.time()
# Define success criteria
success_criteria = define_success_criteria(OBJECTIVE)

print("\033[95m\033[1m" + "\n*****SUCCESS CRITERIA*****\n" + "\033[0m\033[0m")
for s, details in success_criteria.items():
    print(f"{s}: validated={details['validated']}")

# Add the first task
first_task = {"task_id": 1, "task_name": INITIAL_TASK}

add_task(first_task)
# Main loop
task_id_counter = 1
while True:
    if task_list:
        # Print the task list
        print("\033[95m\033[1m" + "\n*****TASK LIST*****\n" + "\033[0m\033[0m")
        for t in task_list:
            print(str(t["task_id"]) + ": " + t["task_name"])

        # Step 1: Pull the first task
        task = task_list.popleft()
        print("\033[92m\033[1m" + "\n*****NEXT TASK*****\n" + "\033[0m\033[0m")
        print(str(task["task_id"]) + ": " + task["task_name"])

        # Send to execution function to complete the task based on the context
        result = execution_agent(OBJECTIVE, task["task_name"])
        this_task_id = int(task["task_id"])
        print("\033[93m\033[1m" + "\n*****TASK RESULT*****\n" + "\033[0m\033[0m")
        print(result)

        # Append the task name and result to the text file
        with open("agent_analysis.txt", "a") as file:
            file.write(f"Task Name: {task['task_name']}\n")
            file.write(f"Result: {result}\n")

        # Step 2: Enrich result and store it in the results_store
        enriched_result = {
            "data": result
        }  # This is where you should enrich the result if needed
        result_id = f"result_{task['task_id']}"
        vector = get_ada_embedding(
            enriched_result["data"]
        )  # get vector of the actual result extracted from the dictionary
        results_store[result_id] = {"vector": vector, "task": task["task_name"], "result": result}


        # Step 3: Create new tasks and reprioritize task list
        new_tasks = task_creation_agent(
            OBJECTIVE,
            enriched_result,
            task["task_name"],
            [t["task_name"] for t in task_list],
            success_criteria,
        )

        for new_task in new_tasks:
            task_id_counter += 1
            new_task.update({"task_id": task_id_counter})
            add_task(new_task)
        prioritization_agent(this_task_id)

        print("\033[95m\033[1m" + "\n*****SUCCESS CRITERIA*****\n" + "\033[0m\033[0m")
        for s, details in success_criteria.items():
            print(f"{s}: validated={details['validated']}")

        if evaluation_agent(OBJECTIVE, results_store, success_criteria):
            print("More than 80pct of success criteria met. Stopping the process.")
            print("\033[95m\033[1m" + "\n*****SUCCESS CRITERIA*****\n" + "\033[0m\033[0m")
            for s, details in success_criteria.items():
                print(f"{s}: validated={details['validated']}")
            break

    time.sleep(1)  # Sleep before checking the task list again

print(f"Time taken to complete the process: {time.time() - start_time:.2f} seconds")
