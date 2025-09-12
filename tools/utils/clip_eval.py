import clip
import torch
from torchvision import transforms
from transformers import AutoImageProcessor, DetrForObjectDetection
#from ldm.models.diffusion.ddim import DDIMSampler
from transformers import AutoImageProcessor, AutoModel, AutoProcessor


class PickEvaluator(object):
    def __init__(self, device) -> None:
        processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        model_pretrained_name_or_path = "yuvalkirstain/PickScore_v1"
        self.device = device

        self.processor = AutoProcessor.from_pretrained(processor_name_or_path)
        self.model = AutoModel.from_pretrained(model_pretrained_name_or_path).eval().to(device)
    
    def calc_probs(self, prompt, images):
        
        # preprocess
        image_inputs = self.processor(
            images=images,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)
        
        text_inputs = self.processor(
            text=prompt,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)


        with torch.no_grad():
            # embed
            image_embs = self.model.get_image_features(**image_inputs)
            image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
        
            text_embs = self.model.get_text_features(**text_inputs)
            text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
        
            # score
            scores = self.model.logit_scale.exp() * (text_embs @ image_embs.T)[0]
            
            # get probabilities if you have multiple images to choose from
            #probs = torch.softmax(scores, dim=-1)
        
        return scores.cpu().numpy()
    
class DINOEvaluator(object):
    def __init__(self, device, dino_model='facebook/dinov2-base'):
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(dino_model)
        self.model = AutoModel.from_pretrained(dino_model).to(device)

    @torch.no_grad()
    def encode_images(self, image):
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        image_features = outputs.last_hidden_state
        image_features = image_features.mean(dim=1)
        
        return image_features

    def get_image_features(self, img, norm=True) -> torch.Tensor:
        image_features = self.encode_images(img)
        if norm:
            image_features /= image_features.clone().norm(dim=-1, keepdim=True)
        return image_features

    def img_to_img_similarity(self, generated_images, src_images=None, src_img_features=None):
        if src_img_features is None and src_images is not None:
            src_img_features = self.get_image_features(src_images)
        gen_img_features = self.get_image_features(generated_images)

        return (src_img_features @ gen_img_features.T).mean()




class CLIPEvaluator(object):
    def __init__(self, device, clip_model='ViT-B/32') -> None:
        self.device = device
        self.model, clip_preprocess = clip.load(clip_model, device=self.device)

        #self.clip_preprocess = clip_preprocess
        
        self.preprocess = clip_preprocess

        # self.preprocess = transforms.Compose([transforms.PILToTensor(),
        #                                       transforms.ConvertImageDtype(torch.float),
        #                                       transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])] + # Un-normalize from [-1.0, 1.0] (generator output) to [0, 1].
        #                                       clip_preprocess.transforms[:2] +                                      # to match CLIP input scale assumptions
        #                                       clip_preprocess.transforms[4:])                                       # + skip convert PIL to tensor

        # 缓存已经生成过的features
        self.text_features_cache = dict()
        self.image_features_cache = dict()

        self.coco_processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
        self.detector = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(device)

    def detect_objects(self, image, prompt):
        if 'and' in prompt.split(' '):
            # 取‘and’后的第一个词
            obj = prompt.split(' and ')[-1]
        elif 'in front of' in prompt:
            obj = prompt.split(' in front of ')[0]
        inputs = self.coco_processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.detector(**inputs)
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.coco_processor.post_process_object_detection(outputs, threshold=0.8, target_sizes=target_sizes)[0]

        for label in results["labels"]:
            if self.detector.config.id2label[label.item()] == obj:
                return 1.
        return 0.

    def tokenize(self, strings: list):
        return clip.tokenize(strings).to(self.device)

    @torch.no_grad()
    def encode_text(self, tokens: list) -> torch.Tensor:
        return self.model.encode_text(tokens)

    @torch.no_grad()
    def encode_images(self, images):
        images = self.preprocess(images).unsqueeze(0).to(self.device)
        return self.model.encode_image(images)

    def get_text_features(self, text: str, norm: bool = True) -> torch.Tensor:

        tokens = clip.tokenize(text).to(self.device)

        text_features = self.encode_text(tokens).detach()

        if norm:
            text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features

    def get_image_features(self, img, norm=True):
        image_features = self.encode_images(img)
        
        if norm:
            image_features /= image_features.clone().norm(dim=-1, keepdim=True)

        return image_features

    def img_to_img_similarity(self, generated_images, src_images=None, src_img_features=None):
        if src_img_features is None and src_images is not None:
            src_img_features = self.get_image_features(src_images)
        gen_img_features = self.get_image_features(generated_images)

        return (src_img_features @ gen_img_features.T).mean()

    def txt_to_img_similarity(self, generated_images, text=None, text_features=None):
        if text_features is None and text is not None:
            text_features = self.get_text_features(text)
        gen_img_features = self.get_image_features(generated_images)

        return (text_features @ gen_img_features.T).mean()
