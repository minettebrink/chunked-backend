from fastapi import FastAPI, HTTPException, Response
from typing import Optional
from pydantic import BaseModel
import torch
from diffusers import LTXImageToVideoPipeline
from diffusers.utils import export_to_video, load_image
from fastapi.middleware.cors import CORSMiddleware
import os
import tempfile

app = FastAPI()

# CORS configuration
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:5173,https://*.koyeb.app"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Model initialization
try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    
    pipe = LTXImageToVideoPipeline.from_pretrained(
        "Lightricks/LTX-Video",
        torch_dtype=dtype,
        cache_dir="/app/model"
    )
    pipe.to(device)
except Exception as e:
    raise RuntimeError(f"Failed to initialize model: {str(e)}")

class VideoGenerationRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    image_url: str
    num_inference_steps: Optional[int] = 50
    width: Optional[int] = 704
    height: Optional[int] = 448
    guidance_scale: Optional[float] = 3.0

@app.post("/generate-video")
async def generate_video(request: VideoGenerationRequest):
    try:
        # Load the image
        image = load_image(request.image_url)

        # Generate video
        video = pipe(
            image=image,
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            num_inference_steps=request.num_inference_steps or 50,
            guidance_scale=request.guidance_scale or 3.0,
            width=request.width,
            height=request.height,
            num_frames=161,
        ).frames[0]

        # Export video to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            export_to_video(video, temp_file.name, fps=24)
            
            with open(temp_file.name, "rb") as video_file:
                video_data = video_file.read()

            return Response(content=video_data, media_type="video/mp4")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating video: {str(e)}")