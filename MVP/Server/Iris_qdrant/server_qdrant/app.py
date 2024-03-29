import logging
import flask
import ai

app = flask.Flask(__name__)
logging_ai: ai.AI = ai.AI()

@app.route("/", methods=["GET"])
def main():
    message: str = flask.request.args.get("message")
    lecture_file: str = flask.request.args.get("lectureFile")
    logging.info(f"Received new request from session 1 with message: {message}")
    return logging_ai.generate_response(message), 200

if __name__ == "__main__":
    logging.info("Starting server...")
    app.run(port=8080)