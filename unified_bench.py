import torch
import clip
import numpy as np
from PIL import Image
import os
import sys
import base64
import mimetypes
import tempfile
from openai import AzureOpenAI
import ImageReward as RM
import hpsv2
from tools.utils.clip_eval import CLIPEvaluator 
from tools.utils.aesthetic_scorer import AestheticScorer as AestheticScorerModel 


class UnifiedScorer:
    def __init__(self, device: str = 'cuda', 
                 gpt_key: str = None, 
                 gpt_endpoint: str = None,
                 gpt_deployment: str = "gpt-4o",
                 verifier_template_path: str = "verifier.txt"):
        
        self.device = device

        print("Loading local models (CLIP, ImageReward, HPSv2, Aesthetic)...")
        self.clip_evaluator = CLIPEvaluator(device=self.device)
        self.image_reward_model = RM.load("ImageReward-v1.0", device=self.device)
        self.aesthetic_scorer = AestheticScorerModel(dtype=torch.float32).to(self.device)
        print("Local models loaded.")

        self.gpt_scorer_enabled = False
        if gpt_key and gpt_endpoint:
            print("Configuring GPT-4o scorer...")
            self.gpt_key = gpt_key
            self.gpt_endpoint = gpt_endpoint
            self.gpt_deployment = gpt_deployment
            try:
                with open(verifier_template_path, "r", encoding="utf-8") as f:
                    self.gpt_template = f.read()
                self.gpt_scorer_enabled = True
                print("GPT-4o scorer configured successfully.")
            except FileNotFoundError:
                print(f"Warning: GPT-4o scorer disabled. Verifier template file not found at: '{verifier_template_path}'")
        else:
            print("Warning: GPT-4o scorer disabled. API key or endpoint not provided.")

    def _encode_image_to_base64(self, image_path: str) -> str:
        mime_type, _ = mimetypes.guess_type(image_path)
        if mime_type is None:
            mime_type = "application/octet-stream"
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:{mime_type};base64,{base64_image}"

    def _calculate_gpt_score(self, prompt: str, image: Image.Image) -> float:
        if not self.gpt_scorer_enabled:
            return -1.0 # 如果未启用，返回一个默认值

        client = AzureOpenAI(
            api_version="2024-02-01",
            azure_endpoint=self.gpt_endpoint,
            api_key=self.gpt_key,
        )
        
        user_textprompt = f"Caption:{prompt} \n Let's think step by step:"
        textprompt = f"{self.gpt_template} \n {user_textprompt}"

        messages = [
            {"role": "system", "content": "You are a helpful assistant and an expert in image analysis."}
        ]

        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as temp_f:
                image.save(temp_f.name, format="PNG")
                print(f"Encoding temporary image for GPT-4o...")
                base64_image_url = self._encode_image_to_base64(temp_f.name)
                
                user_message_content = [
                    {"type": "text", "text": textprompt},
                    {"type": "image_url", "image_url": {"url": base64_image_url}}
                ]
                messages.append({"role": "user", "content": user_message_content})
        except Exception as e:
            print(f"Failure during temporary image creation or encoding: {e}")
            return -1.0

        print("Waiting for Azure GPT-4o response...")
        try:
            response = client.chat.completions.create(
                model=self.gpt_deployment,   
                messages=messages, 
                max_tokens=1024,
                temperature=0.0
            )

            text = response.choices[0].message.content
            print("📝 GPT-4o Raw response:\n", text)
            last_part = text[-10:]
            extracted_score = "".join(filter(str.isdigit, last_part))
            score = int(extracted_score)
            print(f"✅ GPT-4o Extracted score: {score}")
            return float(score)
        except Exception as e:
            print(f"Error during Azure API call: {e}")
            return -1.0 

    def _calculate_clip_score(self, prompt: str, image: Image.Image) -> float:
        score_tensor = self.clip_evaluator.txt_to_img_similarity(generated_images=image, text=prompt)
        return score_tensor.item()

    def _calculate_image_reward(self, prompt: str, image: Image.Image) -> float:
        score_value = self.image_reward_model.score(prompt, [image])
        return float(score_value)

    def _calculate_hps_score(self, prompt: str, image: Image.Image) -> float:
        scores_list = hpsv2.score([image], prompt, hps_version='v2.1')
        return scores_list[0].item()

    def _calculate_aesthetic_score(self, image: Image.Image) -> float:
        img_np = np.array(image.convert('RGB'))
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
        img_tensor = img_tensor.to(self.device, dtype=torch.float32) / 255.0
        
        score_tensor = self.aesthetic_scorer(img_tensor)
        return score_tensor.item()

    def score(self, prompt: str, image: Image.Image) -> dict:
        image_rgb = image.convert("RGB")
        
        print("\nCalculating local scores...")
        scores = {
            "clip_score": self._calculate_clip_score(prompt, image_rgb),
            "image_reward": self._calculate_image_reward(prompt, image_rgb),
            "hps_v2_score": self._calculate_hps_score(prompt, image_rgb),
            "aesthetic_score": self._calculate_aesthetic_score(image_rgb),
        }
        print("Local scores calculated.")
        
        if self.gpt_scorer_enabled:
            print("\nCalculating GPT-4o score...")
            scores["gpt4o_score"] = self._calculate_gpt_score(prompt, image_rgb)
            print("GPT-4o score calculated.")
        
        return scores

if __name__ == '__main__':
    if not torch.cuda.is_available():
        print("Warning: No CUDA GPU detected. Falling back to CPU. This will be very slow.")
        DEVICE = "cpu"
    else:
        DEVICE = "cuda"

    AZURE_API_KEY = os.getenv("AZURE_API_KEY")
    AZURE_API_ENDPOINT = os.getenv("AZURE_API_ENDPOINT")

    if not (AZURE_API_KEY and AZURE_API_ENDPOINT):
        print("\n" + "="*50)
        print("Warning: Environment variables AZURE_API_KEY or AZURE_API_ENDPOINT not set.")
        print("GPT-4o scoring will be disabled.")
        print("To enable it, run the following in your terminal before executing the script:")
        print("  export AZURE_API_KEY='your_key_here'")
        print("  export AZURE_API_ENDPOINT='your_endpoint_here'")
        print("="*50 + "\n")

    print("Initializing UnifiedScorer...")
    try:
        unified_scorer = UnifiedScorer(
            device=DEVICE,
            gpt_key=AZURE_API_KEY,
            gpt_endpoint=AZURE_API_ENDPOINT
        )
    except Exception as e:
        print(f"\nFailed to initialize the scorer: {e}")
        print("Please ensure all dependencies (image-reward, hpsv2, transformers, etc.) are installed correctly.")
        print("Also check that utility scripts (clip_eval.py, aesthetic_scorer.py) are in the correct path.")
        sys.exit(1)
    print("UnifiedScorer initialized successfully.")

    test_prompt = "A green twintail hair girl wearing a white shirt and skirt printed with green apple"
    image_path = "./test.png"

    try:
        test_image = Image.open(image_path)
        print(f"\nSuccessfully loaded test image: '{image_path}'.")
    except FileNotFoundError:
        print(f"'{image_path}' not found. Creating a dummy 1024x1024 black image for demonstration.")
        test_image = Image.new('RGB', (1024, 1024), 'black')

    print("\nCalculating all scores for the image...")
    all_scores = unified_scorer.score(prompt=test_prompt, image=test_image)

    print("\n--- Final Evaluation Scores ---")
    for score_name, value in all_scores.items():
        print(f"{score_name:<20}: {value:.4f} (Type: {type(value).__name__})")