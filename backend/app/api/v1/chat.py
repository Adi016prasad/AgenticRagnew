import boto3
import json
from dotenv import load_dotenv
from botocore.exceptions import ClientError

load_dotenv()

client = boto3.client("bedrock-runtime", region_name="us-west-2")
model_id = "us.meta.llama3-2-3b-instruct-v1:0"

jsonFormat = {
    "output": "vectorDatabaseSearch"
}

conversation_history = [
    {"role": "system", "content": f"You normal talk with user, but if user tells something of pain output in {jsonFormat} for example :- user tells i have pain in neck, your json response `json` {json.dumps(jsonFormat)}"}]

def format_llama3_prompt(history):
    """Converts a list of messages into the Llama 3 prompt format."""
    prompt = "<|begin_of_text|>"
    for entry in history:
        role = entry["role"]
        content = entry["content"]
        prompt += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
    
    prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    return prompt

def ask_llama(user_input):
    conversation_history.append({"role": "user", "content": user_input})
    
    full_prompt = format_llama3_prompt(conversation_history)
    
    native_request = {
        "prompt": full_prompt,
        "max_gen_len": 10,
        "temperature": 0.5,
    }

    try:
        response = client.invoke_model(modelId=model_id, body=json.dumps(native_request))
        model_response = json.loads(response["body"].read())
        response_text = model_response["generation"]
        
        conversation_history.append({"role": "assistant", "content": response_text})
        
        return response_text

    except (ClientError, Exception) as e:
        return f"ERROR: {e}"