from mistralrs import Runner, Which, ChatCompletionRequest

# Using Plain to ensure 100% compatibility with the engine
runner = Runner(
    which=Which.Plain(model_id="Qwen/Qwen3.5-2B"),
    in_situ_quant="4",
)

session_id = "user_123"

# 1. Load session
try:
    with open(f"{session_id}.json", "r") as f:
        runner.import_session(session_id, f.read())
except FileNotFoundError:
    pass

# 2. Inference (using dictionary instead of Message object)
res = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="default",
        messages=[{"role": "user", "content": "Hello! My name is Fabien."}],
        max_tokens=256,
    )
)
print(res.choices[0].message.content)

# 3. Save session
state = runner.export_session(session_id)
if state:
    with open(f"{session_id}.json", "w") as f:
        f.write(state)