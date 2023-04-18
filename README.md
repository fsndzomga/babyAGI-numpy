# BabyAGI-numpy
A version of BabyAGI with numpy instead of pinecone and an evaluation agent to check success criteria

## How to use it
Simple fork this repo, add a .env file with the necessary variables and then run it on your CLI.

The content of your .env file should look like this:

      OPENAI_API_KEY = 'your_openai_api_key'

      OBJECTIVE=write a competitive analysis of the eletrical vehicle market   //(whatever objective you want to achieve)

      INITIAL_TASK=Develop a task list //(you can keep this as is or suggest an initial task to the agent)


Run the script using python3 babyAGI-numpy.py. The AI will begin executing tasks based on the objective and initial task specified in the .env file. It will create, prioritize, and execute tasks using the OpenAI API. The script will continue running until all success criteria are met.

The script will print the current objective, success criteria, task list, next task, task result, and updated success criteria to the console during execution.

## Functionality

The script is built with several functions, each responsible for a different aspect of task management:

add_task(task: Dict): Add a task to the task list.

get_ada_embedding(text): Get an embedding for a text using OpenAI's Ada embedding model.

openai_call(prompt: str, model: str, temperature: float, max_tokens: int): Make an API call to the OpenAI API.

define_success_criteria(objective: str) -> Dict[str, Dict[str, bool]]: Define success criteria based on the objective.

evaluation_agent(objective: str, results_store: Dict, success_criteria: Dict): Evaluate if the success criteria are met.

task_creation_agent(objective: str, result: Dict, task_description: str, task_list: List[str]): Create new tasks based on the result of the previous task.

prioritization_agent(this_task_id: int): Prioritize tasks in the task list.

execution_agent(objective: str, task: str) -> str: Execute a task based on the given objective and previous context.

context_agent(query: str, top_results_num: int): Retrieve relevant context from previous tasks.

