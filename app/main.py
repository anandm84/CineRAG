"""Flask application entry point for CineRAG."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from flask import Flask

from app.routes import register_routes


def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder=str(Path(__file__).parent / "templates"),
        static_folder=str(Path(__file__).parent / "static"),
    )
    app.config["SECRET_KEY"] = "cinerag-dev-key"
    register_routes(app)
    return app


if __name__ == "__main__":
    app = create_app()
    print("Starting CineRAG at http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
