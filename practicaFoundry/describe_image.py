import sys
import base64
import requests
from openai import OpenAI
from io import BytesIO
from PIL import Image

# Local caption pipeline will be loaded lazily to avoid heavy imports when unused
local_caption_pipeline = None

import os
from dotenv import load_dotenv

# Cargar variables desde el archivo .env (busca en el directorio raíz)
load_dotenv()

api_key = os.getenv("AZURE_OPENAI_KEY") or os.getenv("KEY")
base_url = os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("BASE_URL")

def describe_image(image_url: str) -> str:
    try:
        resp = requests.get(image_url, timeout=20)
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "image/jpeg")
        b64 = base64.b64encode(resp.content).decode("utf-8")
        data_uri = f"data:{content_type};base64,{b64}"

        prompt = (
            "A continuación te proporciono una imagen en formato data URI. "
            "Por favor, describe detalladamente lo que aparece en la imagen, incluyendo objetos, contexto y cualquier detalle relevante.\n\n"
            f"IMAGEN: {data_uri}\n\n"
        )

        client = OpenAI(base_url=base_url, api_key=api_key)
        try:
            out = client.responses.create(model="gpt-4o-mini", input=prompt, max_output_tokens=800)
            # If the deployed model refuses to process images, fall back to local captioning
            text = getattr(out, 'output_text', str(out))
            lower = text.lower() if isinstance(text, str) else ''
            # Broad heuristic: if the model indicates it cannot process images/URLs, fallback
            if ('no puedo' in lower or 'cannot' in lower or "can't" in lower or 'no puedo analizar' in lower) and ('imagen' in lower or 'image' in lower or 'url' in lower or 'imagenes' in lower or 'imágenes' in lower):
                # fallback to local caption
                return _local_caption_from_bytes(resp.content)
            return text
        except Exception:
            # On any failure calling remote multimodal, fallback to local caption
            return _local_caption_from_bytes(resp.content)
    except Exception as e:
        return f"ERROR describiendo imagen: {e}"

def main():
    if len(sys.argv) < 2:
        print("Uso: python describe_image.py <image_url>")
        sys.exit(1)
    image_url = sys.argv[1]
    print("Describiendo:", image_url)
    result = describe_image(image_url)
    print("\n--- DESCRIPCIÓN ---\n")
    print(result)

def _local_caption_from_bytes(image_bytes: bytes) -> str:
    """Generate a caption locally using a Hugging Face image-to-text model.
    Requires `transformers`, `torch` and `Pillow` to be installed in the environment.
    """
    global local_caption_pipeline
    try:
        # Use the BLIP processor + model directly for better compatibility
        if local_caption_pipeline is None:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            local_caption_pipeline = {}
            local_caption_pipeline['processor'] = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            local_caption_pipeline['model'] = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        processor = local_caption_pipeline['processor']
        model = local_caption_pipeline['model']
        inputs = processor(images=image, return_tensors="pt")
        out_ids = model.generate(**inputs, max_new_tokens=50, num_beams=3)
        caption = processor.decode(out_ids[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        return f"ERROR en caption local: {e}\n(Instala 'transformers', 'torch' y 'Pillow' en el venv para usar esta función)"

if __name__ == '__main__':
    main()
