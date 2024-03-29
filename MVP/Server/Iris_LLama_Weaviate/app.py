import logging
import flask
import ai

app = flask.Flask(__name__)
logging_ai: ai.AI = ai.AI()


@app.route("/", methods=["GET"])
def main():
    message: str = flask.request.args.get("message")
    return logging_ai.generate_response(message), 200


if __name__ == "__main__":
    logging.info("Starting server...")
    #logging_ai.create_class_llama()
    app.run(port=8080)
