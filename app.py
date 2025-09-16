# from flask import Flask, request, jsonify
# import torch
# import numpy as np
# from config import project_config

# # Load trained model
# from src.models.model import BinaryModel   # your model definition
# device = torch.device("cpu")

# model = BinaryModel(in_channels=5, out_channels=1)
# checkpoint = torch.load(project_config.BASE_DIR / "checkpoints/flood_model_best.pth", map_location=device)
# model.load_state_dict(checkpoint["model_state_dict"])
# model.eval()

# # Create Flask app
# app = Flask(__name__)

# # Home route (browser check)
# @app.route("/", methods=["GET"])
# def home():
#     return "🚀 Flood prediction API is running! Use POST /predict"

# # Prediction route
# @app.route("/predict", methods=["POST"])
# def predict():
#     try:
#         data = request.json
#         x_static = np.array(data["x_static"]).reshape(1, 5, 256, 256)
#         x_rainfall = np.array(data["x_rainfall"]).reshape(1, 1, 256, 256)

#         # Convert to tensors
#         x_static = torch.tensor(x_static, dtype=torch.float32, device=device)
#         x_rainfall = torch.tensor(x_rainfall, dtype=torch.float32, device=device)

#         # Run inference
#         with torch.no_grad():
#             output = model(x_static, x_rainfall).sigmoid().squeeze().cpu().numpy()

#         return jsonify({"prediction": output.tolist()})

#     except Exception as e:
#         return jsonify({"error": str(e)})

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000, debug=True)


from flask import Flask, request, jsonify, send_file
import torch
import numpy as np
import matplotlib.pyplot as plt
import io

# Import your config + model
from config import project_config
from src.models.model import BinaryModel   # <-- your model class

# ==============================
# Model Setup
# ==============================
device = torch.device("cpu")   # use "cuda" if you want GPU

model = BinaryModel(in_channels=5, out_channels=1)
checkpoint = torch.load(
    project_config.BASE_DIR / "checkpoints/flood_model_best.pth",
    map_location=device
)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# ==============================
# Flask App
# ==============================
app = Flask(__name__)

@app.route("/")
def home():
    return "✅ Flood Prediction API is running! Use /predict for JSON or /predict_image for PNG."

# ---------- JSON Prediction ----------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        x_static = np.array(data["x_static"]).reshape(1, 5, 256, 256)
        x_rainfall = np.array(data["x_rainfall"]).reshape(1, 1, 256, 256)

        x_static = torch.tensor(x_static, dtype=torch.float32, device=device)
        x_rainfall = torch.tensor(x_rainfall, dtype=torch.float32, device=device)

        with torch.no_grad():
            output = model(x_static, x_rainfall).sigmoid().squeeze().cpu().numpy()

        return jsonify({"prediction": output.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)})

# ---------- Image Prediction ----------
@app.route("/predict_image", methods=["POST"])
def predict_image():
    try:
        data = request.json
        x_static = np.array(data["x_static"]).reshape(1, 5, 256, 256)
        x_rainfall = np.array(data["x_rainfall"]).reshape(1, 1, 256, 256)

        x_static = torch.tensor(x_static, dtype=torch.float32, device=device)
        x_rainfall = torch.tensor(x_rainfall, dtype=torch.float32, device=device)

        with torch.no_grad():
            output = model(x_static, x_rainfall).sigmoid().squeeze().cpu().numpy()

        # Generate flood map
        fig, ax = plt.subplots()
        ax.imshow(output > 0.5, cmap="Blues")
        ax.set_title("Predicted Flood Map")
        ax.axis("off")

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plt.close(fig)

        return send_file(buf, mimetype="image/png")

    except Exception as e:
        return jsonify({"error": str(e)})

# ==============================
# Run Server
# ==============================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
