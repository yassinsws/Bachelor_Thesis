import logging
import os
import flask
import ai

app = flask.Flask(__name__)
logging_ai: ai.AI = ai.AI()
os.environ["OPENAI_API_KEY"] = ai.azure_openai_key

@app.route("/", methods=["GET"])
def main():
    message: str = flask.request.args.get("message")
    lecture_id: str = flask.request.args.get("lecture_id")
    return logging_ai.generate_response(message, lecture_id), 200

if __name__ == "__main__":
    logging.info("Starting server...")
#    logging_ai.create_class()
    app.run(port=8000)
