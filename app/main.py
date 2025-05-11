from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse
import os
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import CustomObjectScope
from .utils import allowed_file, model_predict, dice_coef, dice_loss

origins = ["*"]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR , 'xception_unet.keras')
with CustomObjectScope({'dice_loss': dice_loss, 'dice_coef': dice_coef}):
    model = load_model(model_path)

templates = Jinja2Templates(directory="app/templates")

@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse(
        request=request, name="index.html"
    )

@app.post('/predict/')
async def predict(file: UploadFile = File(...)):

    if file and allowed_file(file.filename):
        try:
            predicted_mask = model_predict(file, model)  # model must be preloaded
            
            return JSONResponse(status_code=200,content={
                "file": predicted_mask,
                "message": "File uploaded successfully"
            })
        except Exception as e:
            return JSONResponse(status_code=500, content={"message": f"Prediction failed: {str(e)}"})
    else:
        return JSONResponse(
            status_code=400,
            content={"message": "Please upload images of jpg, jpeg, or png extension only"}
        )
