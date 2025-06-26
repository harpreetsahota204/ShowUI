import ast
import os
import logging
import json
from PIL import Image
from typing import Dict, Any, List, Union, Optional
import cv2
from pathlib import Path
import matplotlib.pyplot as plt

import numpy as np
import torch

import fiftyone as fo
from fiftyone import Model, SamplesMixin

from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor, BitsAndBytesConfig
from transformers.utils import is_flash_attn_2_available

from qwen_vl_utils import process_vision_info

logger = logging.getLogger(__name__)

DEFAULT_GROUNDING_SYSTEM_PROMPT = "Based on the screenshot of the page, I give a text description and you give its corresponding location. The coordinate represents a clickable location [x, y] for an element, which is a relative coordinate on the screenshot, scaled from 0 to 1."

DEFAULT_ACTION_GROUNDING_SYSTEM_PROMPT= """You are an assistant trained to navigate the web screen and the phone screen. 
Given a task instruction and a screen observation, you output the next action. Choose the correct action for the web screen or the phone screen. 

Here is the action space:
1. `CLICK`: Click on an element in the web screen, value is not applicable and the position [x,y] is required. 
2. `INPUT`: Type a string into an element, value is a string to type and the position [x,y] is required. 
3. `SELECT`: Select a value for an element, value is not applicable and the position [x,y] is required. 
4. `HOVER`: Hover on an element, value is not applicable and the position [x,y] is required.
5. `ANSWER`: Answer the question, value is the answer and the position is not applicable.
6. `ENTER`: Enter operation, value and position are not applicable.
7. `SCROLL`: Scroll the screen, value is the direction to scroll and the position is not applicable.
8. `SELECT_TEXT`: Select some text content, value is not applicable and position [[x1,y1], [x2,y2]] is the start and end position of the select operation.
9. `COPY`: Copy the text, value is the text to copy and the position is not applicable.
10. `SWIPE`: Swipe the phone screen, value is not applicable and the position [[x1,y1], [x2,y2]] is the start and end position of the swipe operation.
11. `TAP`: Tap on an element on the phone, value is not applicable and the position [x,y] is required.

Format the action as a dictionary with the following keys:
{'action': 'ACTION_TYPE', 'value': 'element', 'position': [x,y]}

If value or position is not applicable, set it as `None`.
Position might be [[x1,y1], [x2,y2]] if the action requires a start and end position.
Position represents the relative coordinates on the screenshot and should be scaled to a range of 0-1.
"""

MIN_PIXELS = 256*28*28
MAX_PIXELS = 1344*28*28

OPERATIONS = {
    "simple_grounding": DEFAULT_GROUNDING_SYSTEM_PROMPT,
    "action_grounding": DEFAULT_ACTION_GROUNDING_SYSTEM_PROMPT,
}

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

class ShowUIModel(SamplesMixin, Model):
    """A FiftyOne model for running ShowUIModel vision tasks"""

    def __init__(
        self,
        model_path: str,
        prompt: str = None,
        operation: str = None,
        system_prompt: str = None,
        quantized: bool = None,
        **kwargs
    ):
        self._fields = {}
        self._operation = operation
        self.model_path = model_path
        self.prompt = prompt
        self.quantized = quantized
        self._custom_system_prompt = system_prompt  # Store custom system prompt if provided
        self.device = get_device()
        logger.info(f"Using device: {self.device}")
        
        # Load model and processor
        logger.info(f"Loading model from {model_path}")

        model_kwargs = {
            "device_map":self.device,
            }

        if is_flash_attn_2_available():
            model_kwargs["attn_implementation"] = "flash_attention_2"

        # Only set specific torch_dtype for CUDA devices
        if self.device == "cuda":
            model_kwargs["torch_dtype"] = torch.bfloat16
            
            # Only apply quantization if device is CUDA
            if self.quantized:
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16
                    )
        elif self.quantized:
            logger.warning("Quantization is only supported on CUDA devices. Ignoring quantization request.")

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            trust_remote_code=True,
            **model_kwargs
        )
        logger.info("Loading processor")

        self.processor = AutoProcessor.from_pretrained(
            model_path, 
            size={
                'shortest_edge': MIN_PIXELS,  # Minimum dimension
                'longest_edge': MAX_PIXELS    # Maximum dimension
                }, 
            use_fast=True
            )

        self.tokenizer = self.processor.tokenizer

        self.model.eval()

    @property
    def operation(self):
        return self._operation

    @operation.setter
    def operation(self, value):
        if value not in OPERATIONS:
            raise ValueError(f"Invalid operation: {value}. Must be one of {list(OPERATIONS.keys())}")
        self._operation = value

    @property
    def system_prompt(self):
        # Return custom system prompt if set, otherwise return default for current operation
        return self._custom_system_prompt if self._custom_system_prompt is not None else OPERATIONS[self.operation]

    @system_prompt.setter
    def system_prompt(self, value):
        self._custom_system_prompt = value

    @property
    def needs_fields(self):
        """A dict mapping model-specific keys to sample field names."""
        return self._fields

    @needs_fields.setter
    def needs_fields(self, fields):
        self._fields = fields
    
    def _get_field(self):
        if "prompt_field" in self.needs_fields:
            prompt_field = self.needs_fields["prompt_field"]
        else:
            prompt_field = next(iter(self.needs_fields.values()), None)

        return prompt_field

    @property
    def media_type(self):
        return "image"

    def _to_keypoints(self, output_text: str, image_width: int, image_height: int) -> fo.Keypoints:
        """Convert model output text to FiftyOne Keypoints."""
        
        if self.operation == "simple_grounding":
                # Parse '[0.14, 0.53]'
                x, y = ast.literal_eval(output_text)
                
                keypoint = fo.Keypoint(
                    label="grounding_point",
                    points=[[x, y]]
                )
            
        elif self.operation == "action_grounding":
            # Parse {'action': 'CLICK', 'value': 'element', 'position': [x,y]}
           
            action_dict = ast.literal_eval(output_text)
            x, y = action_dict['position']
            
            keypoint = fo.Keypoint(
                label=action_dict['action'],
                points=[[x, y]],
                action_value=action_dict['value']
            )
        
        return fo.Keypoints(keypoints=[keypoint])
    
    def _predict(self, image: Image.Image, sample=None) -> fo.Keypoints:
        """Process a single image through the model and return keypoint predictions.
        
        Args:
            image: PIL Image to process
            sample: Optional FiftyOne sample containing the image filepath
            
        Returns:
            fo.Keypoints: Keypoint predictions for GUI interaction points
        """
        # Use local prompt variable instead of modifying self.prompt
        prompt = self.prompt  # Start with instance default
        
        if sample is not None and self._get_field() is not None:
            field_value = sample.get_field(self._get_field())
            if field_value is not None:
                prompt = str(field_value)  # Local variable, doesn't affect instance
        
        if not prompt:
            raise ValueError("No prompt provided.")
        
        messages = [
            {
                "role": "system", 
                "content": [  
                    {
                        "type": "text",
                        "text": self.system_prompt
                    }
                ]
            },
            {
                "role": "user", 
                "content": [
                    {
                        "type": "image", 
                        "image": sample.filepath if sample else image,
                        "min_pixels": MIN_PIXELS, 
                        "max_pixels": MAX_PIXELS
                    },
                    {
                        "type": "text", 
                        "text": prompt
                    },
                ]
            }
        ]

        text = self.processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True,
        )

        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        generated_ids = self.model.generate(
            **inputs, 
            max_new_tokens=128
            )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]

    def predict(self, image, sample=None):
        """Process an image with the model.
        
        Args:
            image: PIL Image or numpy array to process
            sample: Optional FiftyOne sample containing the image filepath
            
        Returns:
            fo.Keypoints: Keypoint predictions for GUI interaction points
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        return self._predict(image, sample)