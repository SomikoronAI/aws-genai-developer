from flask import Flask, request, render_template_string
import boto3
import json

import app_config

app = Flask(__name__)

# Bedrock agent runtime client
bedrock_agent_runtime = boto3.client(service_name="bedrock-agent-runtime", region_name=app_config.region_name)


# HTML template 
HTML = """
<!doctype html>
<title>Auto Insurance Policy Assistant</title>
<h2>Ask an Auto Insurance Policy Question</h2>
<form method="post" action="/ask">
  <input name="question" size="80" required> 
  <input type="hidden" name="temperature" value="0.5">
  <input type="hidden" name="max_tokens" value="256">
  <button type="submit">Ask</button>
</form>

{% if answer %}
  <h3>Answer</h3>
  <p>{{ answer }}</p>

  <form method="post" action="/feedback">
    <input type="hidden" name="question" value="{{ question }}">
    <input type="hidden" name="answer" value="{{ answer }}">
    <button name="rating" value="up">👍 Helpful</button>
    <button name="rating" value="down">👎 Not helpful</button>
  </form>
{% endif %}
"""

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML)

@app.route("/ask", methods=["POST"])
def ask():
    question = request.form["question"]
    temperature = float(request.form.get("temperature", 0.5))
    max_tokens = int(request.form.get("max_tokens", 512))

    response = bedrock_agent_runtime.retrieve_and_generate(
        input={"text": question},
        retrieveAndGenerateConfiguration={
            "type": "KNOWLEDGE_BASE", 
            "knowledgeBaseConfiguration":{
                "knowledgeBaseId": app_config.knowledge_base_id,
                "modelArn": app_config.llm_model_arn,
                "generationConfiguration": {
                    "guardrailConfiguration":{
                        "guardrailId": app_config.guardrail_id, 
                        "guardrailVersion": app_config.guardrail_version
                    },
                    "inferenceConfig": {
                        "textInferenceConfig": {
                            "maxTokens": max_tokens,
                            "temperature": temperature
                        }
                    },
                },
            },
        },
    )
    answer = response["output"]["text"]

    return render_template_string(
        HTML,
        question=question,
        answer=answer
    )


def save_feedback(question, answer, rating):
    feedback = {
        "question": question,
        "answer": answer,
        "rating": rating
    }
    with open("feedback.jsonl", "a") as f:
        f.write(json.dumps(feedback) + "\n")

@app.route("/feedback", methods=["POST"])
def feedback():
    save_feedback(
        request.form["question"],
        request.form["answer"],
        request.form["rating"]
    )
    return "", 204



if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=app_config.debug)