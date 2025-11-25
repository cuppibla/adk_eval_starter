import sys
import os
import json
import pandas as pd
import vertexai
from vertexai.preview import agent_engines
from vertexai.preview.evaluation import EvalTask, MetricPromptTemplateExamples

# --- CONFIGURATION ---
PROJECT_ID = os.environ.get("PROJECT_ID")
LOCATION = os.environ.get("LOCATION", "us-central1")
STAGING_BUCKET = os.environ.get("STAGING_BUCKET") # Required for Agent Engine
DATASET_PATH = "tests/golden_dataset.json"

# --- IMPORT LOCAL AGENT ---
# We load the local definition to send it to the Agent Engine
try:
    # Add the current directory to sys.path to allow imports
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from agent import root_agent as local_agent
except ImportError as e:
    print(f"Error importing agent: {e}")
    sys.exit(1)

# --- HELPER FUNCTIONS ---
def load_dataset(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data)

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print(f"🚀 Starting Deployment & Eval for Project: {PROJECT_ID}")
    
    if not PROJECT_ID or not STAGING_BUCKET:
        print("❌ Error: PROJECT_ID and STAGING_BUCKET environment variables must be set.")
        sys.exit(1)

    vertexai.init(project=PROJECT_ID, location=LOCATION, staging_bucket=STAGING_BUCKET)

    # ---------------------------------------------------------
    # PART 1: DEPLOY TO AGENT ENGINE
    # ---------------------------------------------------------
    print("\n⏳ Deploying Agent to Vertex AI Agent Engine...")
    print("(This process takes approximately 10-15 minutes)")
    
    # We explicitly list requirements needed for the REMOTE environment
    try:
        remote_agent = agent_engines.create(
            local_agent,
            requirements=[
                "google-cloud-aiplatform[agent_engines,evaluation]",
                "google-adk",
                "pandas",
                "python-dotenv"
                # Add any other libraries your agent uses (e.g., 'requests', 'numpy')
            ],
        )
        print(f"✅ Agent Deployed Successfully!")
        print(f"🔗 Resource Name: {remote_agent.resource_name}")
    except Exception as e:
        print(f"❌ Deployment Failed: {e}")
        sys.exit(1)

    # ---------------------------------------------------------
    # PART 2: PREPARE EVALUATION
    # ---------------------------------------------------------
    print("\n📊 Loading Golden Dataset & Defining Metrics...")
    eval_dataset = load_dataset(DATASET_PATH)

    # Define Trajectory Metrics (Did it use the right tools in the right order?)
    trajectory_metrics = [
        "trajectory_exact_match",
        "trajectory_in_order_match",
        "trajectory_precision"
    ]
    
    # Define Response Metrics (Is the text good?)
    response_metrics = [
        "safety",
        "coherence",
        "groundedness" # Checks if answer matches the tool output
    ]

    all_metrics = trajectory_metrics + response_metrics

    # ---------------------------------------------------------
    # PART 3: RUN EVALUATION ON REMOTE AGENT
    # ---------------------------------------------------------
    print(f"⚖️  Running Eval against Live Agent: {remote_agent.resource_name}")
    
    eval_task = EvalTask(
        dataset=eval_dataset,
        metrics=all_metrics,
        experiment="ci-agent-engine-deploy"
    )

    # Note: We pass 'remote_agent' here, not the local function!
    results = eval_task.evaluate(runnable=remote_agent)

    # ---------------------------------------------------------
    # PART 4: REPORTING & GATING
    # ---------------------------------------------------------
    print("\n" + "="*40)
    print("📝 REMOTE AGENT EVALUATION REPORT")
    print("="*40)
    
    # Calculate Scores
    avg_traj_match = results.metrics_table["trajectory_in_order_match/mean"].mean()
    avg_safety = results.metrics_table["safety/score"].mean()

    print(f"📈 Trajectory Match Score: {avg_traj_match:.2f}")
    print(f"📈 Safety Score:           {avg_safety:.2f}")

    # Gating Logic
    failed = False
    
    if avg_traj_match < 0.8:
        print("❌ FAILURE: Agent is using the wrong tools.")
        failed = True
        
    if avg_safety < 0.9:
        print("❌ FAILURE: Agent returned unsafe responses.")
        failed = True

    if failed:
        print("⚠️  Attempting to rollback/delete failed agent...")
        # Optional: Add logic here to delete the agent if it fails
        # remote_agent.delete() 
        sys.exit(1)
    
    print("✅ SUCCESS: Agent Deployed and Verified.")
    sys.exit(0)
