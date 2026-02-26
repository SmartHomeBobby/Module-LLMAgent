import os
import json
import uuid
import time
import threading
import builtins
from datetime import datetime
import paho.mqtt.client as mqtt
from dotenv import load_dotenv
import ollama

original_print = builtins.print

def timestamped_print(*args, **kwargs):
    if not args and not kwargs:
        original_print()
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        original_print(f"[{timestamp}]", *args, **kwargs)

builtins.print = timestamped_print

# Load environment variables
load_dotenv()

MQTT_BROKER = os.getenv("MQTT_BROKER", "127.0.0.1")
MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))
MQTT_USER = os.getenv("MQTT_USER", "")
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD", "")
MQTT_TOPIC_REQUEST = os.getenv("MQTT_TOPIC_REQUEST", "smarthomebobby/llm/request")
MQTT_TOPIC_RESPONSE = os.getenv("MQTT_TOPIC_RESPONSE", "smarthomebobby/llm/response")
AGENT_HOST_NAME = os.getenv("AGENT_HOST_NAME", "local-python-agent")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

if OLLAMA_BASE_URL:
    os.environ["OLLAMA_HOST"] = OLLAMA_BASE_URL

MODEL_MAPPING = {
    "General": "haervwe/GLM-4.6V-Flash-9B",
    0: "haervwe/GLM-4.6V-Flash-9B",
    "CodeGeneration": "qwen2.5-coder:32b",
    1: "qwen2.5-coder:32b",
    "CodeExplanation": "deepseek-coder-v2:16b",
    2: "deepseek-coder-v2:16b",
}

DEFAULT_MODEL = "haervwe/GLM-4.6V-Flash-9B"


def pull_model_if_missing(model_name: str):
    try:
        print(f"Checking model {model_name}...")
        models = [m['model'] for m in ollama.list()['models']]
        if not any(model_name in m for m in models):
            print(f"Pulling {model_name}...")
            ollama.pull(model_name)
        print(f"Model {model_name} ready.")
    except Exception as e:
        print(f"Model error: {e}")


def create_response_event(trace_id: str, priority: int, response_text: str, elapsed_seconds: float) -> dict:
    h, rem = divmod(elapsed_seconds, 3600)
    m, s = divmod(rem, 60)
    time_span_str = f"{int(h):02d}:{int(m):02d}:{s:011.7f}"
    return {
        "TraceId": trace_id,
        "EventId": str(uuid.uuid4()),
        "CreationTime": datetime.utcnow().isoformat() + "Z",
        "Sender": {"Module": "llm-agent-python", "Host": AGENT_HOST_NAME, "Version": "1.0.0"},
        "Priority": priority,
        "Response": response_text,
        "GenerationTime": time_span_str
    }


# Global client for thread-safety
client = None
active_requests = 0
active_requests_lock = threading.Lock()


def handle_request(payload):
    """Run in separate thread to unblock MQTT loop."""
    global client, active_requests
    
    with active_requests_lock:
        active_requests += 1
        current_active = active_requests
        
    trace_id = payload.get("TraceId", payload.get("traceId", str(uuid.uuid4())))
    print(f"[{trace_id}] Started processing. Total active LLM requests: {current_active}")

    try:
        request_text = payload.get("Request", payload.get("request", ""))
        if not request_text:
            print("No request text")
            return
        request_type = payload.get(
            "RequestType", payload.get("requestType", 0))
        priority = payload.get("Priority", payload.get("priority", 2))

        target_model = MODEL_MAPPING.get(request_type, DEFAULT_MODEL)
        print(f"[{trace_id}] Using {target_model}")
        pull_model_if_missing(target_model)

        start_time = time.time()
        print(f"[{trace_id}] Generating response...")
        
        import sys
        
        response_stream = ollama.generate(
            model=target_model, 
            prompt=request_text, 
            options={"keep_alive": -1},
            stream=True
        )
        
        response_text = ""
        for chunk in response_stream:
            text_chunk = chunk.get("response", "")
            if text_chunk:
                response_text += text_chunk
                sys.stdout.write(text_chunk)
                sys.stdout.flush()
        
        print() # New line after the streamed response
        elapsed = time.time() - start_time

        print(f"[{trace_id}] Done in {elapsed:.2f}s")
        event = json.dumps(create_response_event(
            trace_id, priority, response_text, elapsed))
        result, mid = client.publish(
            MQTT_TOPIC_RESPONSE, event, qos=2)
        print(f"[{trace_id}] Published mid={mid} rc={result}")
    except Exception as e:
        print(f"[{payload.get('EventId', 'unknown')}] Error: {e}")
    finally:
        with active_requests_lock:
            active_requests -= 1
            remaining = active_requests
        print(f"[{trace_id}] Finished processing. Remaining active LLM requests: {remaining}")


def on_connect(client_local, userdata, flags, rc, properties=None):
    print(f"Connected rc={rc}")
    if rc == 0:
        client_local.subscribe(MQTT_TOPIC_REQUEST, qos=2)
        print(f"Subscribed to topic: {MQTT_TOPIC_REQUEST} with QoS 2")


def on_message(client_local, userdata, msg):
    print(f"Msg on {msg.topic}")
    try:
        payload = json.loads(msg.payload.decode())
        print(f"Req ID: {payload.get('EventId', 'unknown')}")
        # Offload to thread
        thread = threading.Thread(
            target=handle_request, args=(payload,), daemon=True)
        thread.start()
    except Exception as e:
        print(f"Msg error: {e}")


def on_publish(client_local, userdata, mid, reason_code=0, properties=None):
    print(f"Published mid={mid} code={reason_code}")


def on_log(client_local, userdata, level, buf):
    print(f"LOG {level}: {buf}")


def main():
    global client
    from paho.mqtt.client import CallbackAPIVersion
    
    client_id = f"llm_agent_{AGENT_HOST_NAME}"
    
    try:
        # paho-mqtt 2.x interface
        client = mqtt.Client(CallbackAPIVersion.VERSION2, client_id=client_id, clean_session=False)
    except:
        # paho-mqtt 1.x interface
        client = mqtt.Client(client_id=client_id, clean_session=False)

    client.on_connect = on_connect
    client.on_message = on_message  # Correct sig: no properties
    client.on_publish = on_publish
    # client.on_log = on_log

    if MQTT_USER and MQTT_PASSWORD:
        client.username_pw_set(MQTT_USER, MQTT_PASSWORD)

    print("Connecting...")
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_forever()


if __name__ == "__main__":
    main()
