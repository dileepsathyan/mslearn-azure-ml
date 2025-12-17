# Install Python SDK
pip install azure-ai-ml



# Create a Workspace

from azure.ai.ml.entities import Workspace

workspace_name = "mlw-example"

ws_basic = Workspace(
    name=workspace_name,
    location="eastus",
    display_name="Basic workspace-example",
    description="This example shows how to create a basic workspace",
)
ml_client.workspaces.begin_create(ws_basic)



# Config file to contain 2 parameters:
{
    "subscription_id": "38245d22-4620-4106-9d08-a07ed74df31e",
    "resource_group": "rg-dp100-lab11",
    "workspace_name": "mlw-dp100-lab11"
}


# Connecting to an Existing Workspace

from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azure.ai.ml import MLClient

try:
    credential = DefaultAzureCredential()
    # Check if given credential can get token successfully.
    credential.get_token("https://management.azure.com/.default")
except Exception as ex:
    # Fall back to InteractiveBrowserCredential in case DefaultAzureCredential not work
    credential = InteractiveBrowserCredential()



# Get a handle to workspace
ml_client = MLClient.from_config(credential=credential)



# Running a job
from azure.ai.ml import command

job = command(
    code="./src",
    command="python train.py",
    environment="AzureML-sklearn-0.24-ubuntu18.04-py37-cpu@latest",
    compute="aml-cluster",
    experiment_name="train-model"
)
# connect to workspace and submit job
returned_job = ml_client.create_or_update(job)




# Create URI file data asset
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes

my_path = '<supported-path>'

my_data = Data(
    path=my_path,
    type=AssetTypes.URI_FILE,
    description="<description>",
    name="<name>",
    version="<version>")
ml_client.data.create_or_update(my_data)

# You can read the data by including the following code in your Python script:
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--input_data", type=str)
args = parser.parse_args()

df = pd.read_csv(args.input_data)
print(df.head(10))





# Create a URI folder data asset
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes

my_path = '<supported-path>'

my_data = Data(
    path=my_path,
    type=AssetTypes.URI_FOLDER,
    description="<description>",
    name="<name>",
    version='<version>')
ml_client.data.create_or_update(my_data)

# You can read all CSV files in the folder and concatenate them by
import argparse
import glob
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--input_data", type=str)
args = parser.parse_args()

data_path = args.input_data
all_files = glob.glob(data_path + "/*.csv")
df = pd.concat((pd.read_csv(f) for f in all_files), sort=False)




# Create a MLTable data asset
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes

my_path = '<path-including-mltable-file>'

my_data = Data(
    path=my_path,
    type=AssetTypes.MLTABLE,
    description="<description>",
    name="<name>",
    version='<version>')
ml_client.data.create_or_update(my_data)

# Read the MLTable data using:
import argparse
import mltable
import pandas

parser = argparse.ArgumentParser()
parser.add_argument("--input_data", type=str)
args = parser.parse_args()

tbl = mltable.load(args.input_data)
df = tbl.to_pandas_dataframe()

print(df.head(10))




# Create a Compute instance
from azure.ai.ml.entities import ComputeInstance

ci_basic_name = "basic-ci-12345"
ci_basic = ComputeInstance(
    name=ci_basic_name, 
    size="STANDARD_DS3_v2"
)
ml_client.begin_create_or_update(ci_basic).result()




# Create a Compute Cluster
from azure.ai.ml.entities import AmlCompute

cluster_basic = AmlCompute(
    name="cpu-cluster",
    type="amlcompute",         # Important
    size="STANDARD_DS3_v2",
    location="westus",
    min_instances=0,
    max_instances=2,            # Important
    idle_time_before_scale_down=120,
    tier="low_priority",        # Important (low_priority or dedicated)
)
ml_client.begin_create_or_update(cluster_basic).result()






# Push as job command
from azure.ai.ml import command

# configure job
job = command(
    code="./src",
    command="python diabetes-training.py",
    environment="AzureML-sklearn-0.24-ubuntu18.04-py37-cpu@latest",
    compute="cpu-cluster",
    display_name="train-with-cluster",
    experiment_name="diabetes-training")

# submit job
returned_job = ml_client.create_or_update(job)
aml_url = returned_job.studio_url
print("Monitor your job at", aml_url)





# List and view the available environments in Azure Container Registry
envs = ml_client.environments.list()
for env in envs:
    print(env.name)


# To review the details of a specific environment
env = ml_client.environments.get(name="my-environment", version="1")
print(env)




# Creating a custom environment from Docker image
#YAML file
(
    name: basic-env-cpu
channels:
  - conda-forge
dependencies:
  - python=3.7
  - scikit-learn
  - pandas
  - numpy
  - matplotlib
)

from azure.ai.ml.entities import Environment

env_docker_conda = Environment(
    image="mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04",
    conda_file="./conda-env.yml",
    name="docker-image-plus-conda-example",
    description="Environment created from a Docker image plus Conda environment.",
)
ml_client.environments.create_or_update(env_docker_conda)





# Configure an AutoML experiment job
from azure.ai.ml import automl

# configure the classification job
classification_job = automl.classification(
    compute="aml-cluster",
    experiment_name="auto-ml-class-dev",
    training_data=my_training_data_input,
    target_column_name="Diabetic",
    primary_metric="accuracy",          # Important
    n_cross_validations=5,
    enable_model_explainability=True)

# Set limits
classification_job.set_limits(
    timeout_minutes=60,             # of minutes after which the complete AutoML experiment is terminated.
    trial_timeout_minutes=20,       # Maximum number of minutes one trial can take.
    max_trials=5,                   # Maximum number of trials, or models that will be trained.
    enable_early_termination=True,  # to end the experiment if the score isn't improving in the short term.
)

# submit the AutoML job
returned_job = ml_client.jobs.create_or_update(classification_job)

# Monitor the AutoML job
aml_url = returned_job.studio_url
print("Monitor your job at", aml_url)





# Enable auto-logging for an MLflow experiment
from xgboost import XGBClassifier

with mlflow.start_run():
    mlflow.xgboost.autolog()

    model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)




# Custom logging with MLflow
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

with mlflow.start_run():
    model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", accuracy)






# Search Space
from azure.ai.ml.sweep import Choice, Normal

command_job_for_sweep = job(
    batch_size=Choice(values=[16, 32, 64]),    
    learning_rate=Normal(mu=10, sigma=3),)






# Sampling Methods

# Grid sampling: Tries every possible combination.
from azure.ai.ml.sweep import Choice

command_job_for_sweep = command_job(
    batch_size=Choice(values=[16, 32, 64]),
    learning_rate=Choice(values=[0.01, 0.1, 1.0]),)

sweep_job = command_job_for_sweep.sweep(sampling_algorithm = "grid",)



# Random sampling: Randomly selects combinations.
from azure.ai.ml.sweep import Normal, Uniform

command_job_for_sweep = command_job(
    batch_size=Choice(values=[16, 32, 64]),   
    learning_rate=Normal(mu=10, sigma=3),)

sweep_job = command_job_for_sweep.sweep(sampling_algorithm = "random",)


# Sobol sampling: a type of random sampling that allows you to use a seed. 
# When you add a seed, the sweep job can be reproduced, and the search space distribution is spread more evenly.
from azure.ai.ml.sweep import RandomSamplingAlgorithm

sweep_job = command_job_for_sweep.sweep(
    sampling_algorithm = RandomSamplingAlgorithm(seed=123, rule="sobol"),)


# Bayesian sampling: Uses results from prior runs to inform the selection of hyperparameter values for future runs.
from azure.ai.ml.sweep import Uniform, Choice

command_job_for_sweep = job(
    batch_size=Choice(values=[16, 32, 64]),    
    learning_rate=Uniform(min_value=0.05, max_value=0.1),)

sweep_job = command_job_for_sweep.sweep(sampling_algorithm = "bayesian",)






# Create a training script for hyperparameter tuning:
# following example script trains a logistic regression model using a --regularization argument 
# to set the regularization rate hyperparameter, and logs the accuracy metric with the name Accuracy:
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import mlflow

# get regularization hyperparameter
parser = argparse.ArgumentParser()
parser.add_argument('--regularization', type=float, dest='reg_rate', default=0.01)
args = parser.parse_args()
reg = args.reg_rate

# load the training dataset
data = pd.read_csv("data.csv")

# separate features and labels, and split for training/validatiom
X = data[['feature1','feature2','feature3','feature4']].values
y = data['label'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

# train a logistic regression model with the reg hyperparameter
model = LogisticRegression(C=1/reg, solver="liblinear").fit(X_train, y_train)

# calculate and log accuracy
y_hat = model.predict(X_test)
acc = np.average(y_hat == y_test)
mlflow.log_metric("Accuracy", acc)      # Logs Accuracy metric


# configure command job as base
from azure.ai.ml import command
from azure.ai.ml.sweep import Choice


job = command(
    code="./src",
    command="python train.py --regularization ${{inputs.reg_rate}}",
    inputs={"reg_rate": 0.01,},
    environment="AzureML-sklearn-0.24-ubuntu18.04-py37-cpu@latest",
    compute="aml-cluster", 
    )

# override input for hyperparameter sweep
command_job_for_sweep = job(reg_rate=Choice(values=[0.01, 0.1, 1]),)

# Finally, call sweep() on your command job to sweep over your search space:
from azure.ai.ml import MLClient

# apply the sweep parameter to obtain the sweep_job
sweep_job = command_job_for_sweep.sweep(
    compute="aml-cluster",
    sampling_algorithm="grid",
    primary_metric="Accuracy",
    goal="Maximize",)

# set the name of the sweep job experiment
sweep_job.experiment_name="sweep-example"

# define the limits for this sweep
sweep_job.set_limits(max_total_trials=4, max_concurrent_trials=2, timeout=7200)

# submit the sweep
returned_sweep_job = ml_client.create_or_update(sweep_job)






# Create a component: To create a component for the prep.py script, you'll need a YAML file prep.yml:
(
$schema: https://azuremlschemas.azureedge.net/latest/commandComponent.schema.json
name: prep_data
display_name: Prepare training data
version: 1
type: command
inputs:
  input_data: 
    type: uri_file
outputs:
  output_data:
    type: uri_file
code: ./src
environment: azureml:AzureML-sklearn-0.24-ubuntu18.04-py37-cpu@latest
command: >-
  python prep.py 
  --input_data ${{inputs.input_data}}
  --output_data ${{outputs.output_data}}
)

# YAML file refers to the prep.py script, which is stored in the src folder. 
# You can load the component with the following code:
from azure.ai.ml import load_component
parent_dir = ""

loaded_component_prep = load_component(source=parent_dir + "./prep.yml")

# You can register a component with the following code:
prep = ml_client.components.create_or_update(prepare_data_component)







# Build a pipeline: create the YAML file, or use the @pipeline() function to create the YAML file.
from azure.ai.ml.dsl import pipeline

@pipeline()
def pipeline_function_name(pipeline_job_input):
    prep_data = loaded_component_prep(input_data=pipeline_job_input)
    train_model = loaded_component_train(training_data=prep_data.outputs.output_data)

    return {
        "pipeline_job_transformed_data": prep_data.outputs.output_data,
        "pipeline_job_trained_model": train_model.outputs.model_output,
    }


# pass a registered data asset as the pipeline job input
from azure.ai.ml import Input
from azure.ai.ml.constants import AssetTypes

pipeline_job = pipeline_function_name(
    Input(type=AssetTypes.URI_FILE, 
    path="azureml:data:1"))



# Schedule a pipeline job
# frequency setting for the pipeline job
from azure.ai.ml.entities import RecurrenceTrigger

schedule_name = "run_every_minute"

recurrence_trigger = RecurrenceTrigger(
    frequency="minute",         # minute, hour, day, week, month
    interval=1,)


# schedule the job
from azure.ai.ml.entities import JobSchedule

job_schedule = JobSchedule(name=schedule_name, 
                           trigger=recurrence_trigger, 
                           create_job=pipeline_job)

job_schedule = ml_client.schedules.begin_create_or_update(
    schedule=job_schedule).result()



# Disable and Delete the pipeline job schedule
ml_client.schedules.begin_disable(name=schedule_name).result()
ml_client.schedules.begin_delete(name=schedule_name).result()





# MLmodel YML format example: the important things are flavor and signature.
(
artifact_path: classifier
flavors:
  fastai:
    data: model.fastai
    fastai_version: 2.4.1
  python_function:
    data: model.fastai
    env: conda.yaml
    loader_module: mlflow.fastai
    python_version: 3.8.12
model_uuid: e694c68eba484299976b06ab9058f636
run_id: e13da8ac-b1e6-45d4-a9b2-6a0a5cfac537
signature:
  inputs : '[{"type": "tensor", "tensor-spec": {"dtype": "uint8", "shape": [-1, 300, 300, 3]}}]'
  outputs: '[{"type": "tensor", "tensor-spec": {"dtype": "float32", "shape": [-1,2]}}]'
)

# example of the Python function flavor may look like:
(
artifact_path: pipeline
flavors:
  python_function:
    env:
      conda: conda.yaml
      virtualenv: python_env.yaml
    loader_module: mlflow.sklearn
    model_path: model.pkl
    predict_fn: predict
    python_version: 3.8.5
  sklearn:
    code: null
    pickled_model: model.pkl
    serialization_format: cloudpickle
    sklearn_version: 1.2.0
mlflow_version: 2.1.0
model_uuid: b8f9fe56972e48f2b8c958a3afb9c85d
run_id: 596d2e7a-c7ed-4596-a4d2-a30755c0bfa5
signature:
  inputs: '[{"name": "age", "type": "long"}, {"name": "sex", "type": "long"}, {"name":
    "cp", "type": "long"}, {"name": "trestbps", "type": "long"}, {"name": "chol",
    "type": "long"}, {"name": "fbs", "type": "long"}, {"name": "restecg", "type":
    "long"}, {"name": "thalach", "type": "long"}, {"name": "exang", "type": "long"},
    {"name": "oldpeak", "type": "double"}, {"name": "slope", "type": "long"}, {"name":
    "ca", "type": "long"}, {"name": "thal", "type": "string"}]'
  outputs: '[{"name": "target", "type": "long"}]'
)






# Create an endpoint:
from azure.ai.ml.entities import ManagedOnlineEndpoint

# create an online endpoint
endpoint = ManagedOnlineEndpoint(
    name="endpoint-example",            # name of the endpoint
    description="Online endpoint",      
    auth_mode="key",)                   # key (for key-based auth) or aml-token (for Azure token-based auth)

ml_client.begin_create_or_update(endpoint).result()





# To deploy (and automatically register) the model, run the following command:
from azure.ai.ml.entities import Model, ManagedOnlineDeployment
from azure.ai.ml.constants import AssetTypes

# create a blue deployment
model = Model(
    path="./model",
    type=AssetTypes.MLFLOW_MODEL,
    description="my sample mlflow model",
)

blue_deployment = ManagedOnlineDeployment(
    name="blue",
    endpoint_name="endpoint-example",
    model=model,
    instance_type="Standard_F4s_v2",        # VM size
    instance_count=1,)                      # number of instances

ml_client.online_deployments.begin_create_or_update(blue_deployment).result()

# blue deployment takes 100 traffic
endpoint.traffic = {"blue": 100}
ml_client.begin_create_or_update(endpoint).result()


# To delete the endpoint and all associated deployments, run the following command:
ml_client.online_endpoints.begin_delete(name="endpoint-example")







# Deploy a model to an online endpoint:
import json
import joblib
import numpy as np
import os

# called when the deployment is created or updated
def init():
    global model
    # get the path to the registered model file and load it
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model.pkl')
    model = joblib.load(model_path)

# called when a request is received
def run(raw_data):
    # get the input data as a numpy array
    data = np.array(json.loads(raw_data)['data'])
    # get a prediction from the model
    predictions = model.predict(data)
    # return the predictions as any JSON serializable format
    return predictions.tolist()





# Create an environment for the deployment:
conda.yml file:
(
    name: basic-env-cpu
    channels:
    - conda-forge
    dependencies:
    - python=3.7
    - scikit-learn
    - pandas
    - numpy
    - matplotlib
)

# to create the environment, run the following code:
from azure.ai.ml.entities import Environment

env = Environment(
    image="mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04",
    conda_file="./src/conda.yml",
    name="deployment-environment",
    description="Environment created from a Docker image plus Conda environment.",)

ml_client.environments.create_or_update(env)


# Testing the deployment
# test the blue deployment with some sample data
response = ml_client.online_endpoints.invoke(
    endpoint_name=online_endpoint_name,
    deployment_name="blue",
    request_file="sample-data.json",)

if response[1]=='1':
    print("Yes")
else:
    print ("No")






# Create a batch endpoint:
from azure.ai.ml.entities import BatchEndpoint

endpoint = BatchEndpoint(
    name="endpoint-example",
    description="A batch endpoint",)

ml_client.batch_endpoints.begin_create_or_update(endpoint)


# register the model as MLflow model
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes

model_name = 'mlflow-model'
model = ml_client.models.create_or_update(
            Model(name=model_name, 
                  path='./model', 
                  type=AssetTypes.MLFLOW_MODEL))


# Deploy a batch deployment:
from azure.ai.ml.entities import BatchDeployment, BatchRetrySettings
from azure.ai.ml.constants import BatchDeploymentOutputAction

deployment = BatchDeployment(
    name="forecast-mlflow",
    description="A sales forecaster",
    endpoint_name=endpoint.name,
    model=model,
    compute="aml-cluster",
    instance_count=2,                   # number of compute nodes
    max_concurrency_per_instance=2,     # number of parallel tasks per node
    mini_batch_size=2,                  # number of files per batch
    output_action=BatchDeploymentOutputAction.APPEND_ROW,       # how to store output
    output_file_name="predictions.csv",                         # output file name
    retry_settings=BatchRetrySettings(max_retries=3, timeout=300),      # retry settings
    logging_level="info",)              # logging level

ml_client.batch_deployments.begin_create_or_update(deployment)





# Responsible AI
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from openai import AzureOpenAI

try:
    
    # connect to the project
    project_endpoint = "https://......"
    project_client = AIProjectClient(            
            credential=DefaultAzureCredential(),
            endpoint=project_endpoint,)
    
    # Get a chat client
    chat_client = project_client.get_openai_client(api_version="2024-10-21")
    
    # Get a chat completion based on a user-provided prompt
    user_prompt = input("Enter a question:")
    
    response = chat_client.chat.completions.create(
        model=your_model_deployment_name,
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": user_prompt}
        ])
    print(response.choices[0].message.content)

except Exception as ex:
    print(ex)






# Create a RAG-based client application
from openai import AzureOpenAI

# Get an Azure OpenAI chat client
chat_client = AzureOpenAI(
    api_version = "2024-12-01-preview",
    azure_endpoint = open_ai_endpoint,
    api_key = open_ai_key
)

# Initialize prompt with system message
prompt = [{"role": "system", 
           "content": "You are a helpful AI assistant."}]

# Add a user input message to the prompt
input_text = input("Enter a question: ")
prompt.append({"role": "user", "content": input_text})

# Additional parameters to apply RAG pattern using the AI Search index
rag_params = {
    "data_sources": [
        {   "type": "azure_search",
            "parameters": {
                "endpoint": search_url,
                "index_name": "index_name",
                "authentication": {
                    "type": "api_key",
                    "key": search_key,
                }}}],}

# Submit the prompt with the index information
response = chat_client.chat.completions.create(
    model="<model_deployment_name>",
    messages=prompt,
    extra_body=rag_params)

# Print the contextualized response
completion = response.choices[0].message.content
print(completion)




# To use a vector-based query, you can modify the specification of the Azure AI Search data source details to include an embedding model; which is then used to vectorize the query text.
rag_params = {
    "data_sources": [
        {   "type": "azure_search",
            "parameters": {
                "endpoint": search_url,
                "index_name": "index_name",
                "authentication": {
                    "type": "api_key",
                    "key": search_key,
                },
                # Params for vector-based query
                "query_type": "vector",
                "embedding_dependency": {
                    "type": "deployment_name",
                    "deployment_name": "<embedding_model_deployment_name>",
                },}}],}
