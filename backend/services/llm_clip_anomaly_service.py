import os
import cv2
import numpy as np
import base64
import uuid
import json
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import requests
from datetime import datetime

class LLMCLIPAnomalyService:
    """
    Implementation of the research paper approach:
    LLM-Generated Prompts + GroundingDINO + CLIP for Zero-Shot Anomaly Detection
    
    Workflow:
    1. LLM generates normal and anomaly descriptions for given component/context
    2. GroundingDINO localizes objects using LLM-generated prompts
    3. CLIP encodes both the localized image regions and text descriptions
    4. Anomaly detection via cosine similarity between image and text embeddings
    5. Classification of anomalies using semantic similarity
    """
    
    def __init__(self):
        self.results_dir = os.path.abspath(os.path.join("ml", "results", "llm_clip_anomaly"))
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Initialize models
        self._clip_model = None
        self._clip_processor = None
        self._llm_client = None
        
        # Import services
        from backend.services.grounding_dino_service import grounding_dino_service
        self.grounding_dino = grounding_dino_service
        
        # Local LLM pipeline (will be loaded on demand)
        self._local_llm_pipeline = None
        
        print("🧠 LLM + GroundingDINO + CLIP service initialized")
    
    def _ensure_clip_loaded(self) -> bool:
        """Load CLIP model if not already loaded"""
        if self._clip_model is not None:
            return True
            
        try:
            print("📦 Loading CLIP model...")
            from transformers import CLIPModel, CLIPProcessor
            import torch
            
            # Use a lightweight CLIP model for faster inference
            model_name = "openai/clip-vit-base-patch32"
            
            self._clip_processor = CLIPProcessor.from_pretrained(model_name)
            self._clip_model = CLIPModel.from_pretrained(model_name)
            
            # Auto-detect device
            if torch.cuda.is_available():
                self._clip_model = self._clip_model.cuda()
                print("🚀 CLIP loaded on GPU")
            else:
                print("💻 CLIP loaded on CPU")
            
            self._clip_model.eval()
            return True
            
        except ImportError as e:
            print(f"❌ CLIP not available: {e}")
            print("Install with: pip install transformers torch")
            return False
        except Exception as e:
            print(f"❌ Error loading CLIP: {e}")
            return False
    
    def _ensure_llm_client(self) -> bool:
        """Initialize LLM client (Local HuggingFace, OpenRouter, OpenAI, or fallback)"""
        if self._llm_client is not None or self._local_llm_pipeline is not None:
            return True
        
        try:
            # Try local HuggingFace model first (completely free)
            if self._try_local_llm():
                return True
            
            # Try OpenRouter (often cheaper/more accessible)
            if self._try_openrouter_client():
                return True
            
            # Fallback to OpenAI
            if self._try_openai_client():
                return True
            
            print("⚠️ No LLM available. Using template fallback.")
            return False
            
        except Exception as e:
            print(f"⚠️ LLM client setup failed: {e}")
            return False
    
    def _try_local_llm(self) -> bool:
        """Try to initialize local HuggingFace LLM"""
        try:
            from transformers import pipeline
            import torch
            
            print("🤖 Loading local LLM (this may take a few minutes on first run)...")
            
            # Use small, efficient models (ordered by size - smallest first)
            model_options = [
                # "distilgpt2",                    # 82MB - Very fast and lightweight
                # "gpt2",                          # 500MB - Reliable baseline
                # "microsoft/DialoGPT-medium",
                "Qwen/Qwen2.5-0.5B-Instruct",   # 1GB - Smaller Qwen model
                "Qwen/Qwen3-4B-Instruct-2507",  # 8GB+ - Currently loaded but slow on CPU
                "Qwen/Qwen3-4B-Instruct-2507",  # 8GB+ - Currently loaded but slow on CPU
            ]
            
            for model_id in model_options:
                try:
                    print(f"📦 Trying to load: {model_id}")
                    
                    # Auto-detect device
                    device = 0 if torch.cuda.is_available() else -1
                    device_name = "GPU" if device == 0 else "CPU"
                    
                    # Set memory and timeout constraints
                    pipeline_kwargs = {
                        "model": model_id,
                        "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
                        "device": device,
                        "max_length": 256,  # Reduced context to save memory
                        "do_sample": True,
                        "temperature": 0.7,
                    }
                    
                    # Add pad token for GPT-based models
                    if "gpt" in model_id.lower():
                        pipeline_kwargs["pad_token_id"] = 50256
                    
                    # Add memory optimization for larger models
                    if "qwen" in model_id.lower():
                        pipeline_kwargs["model_kwargs"] = {
                            "low_cpu_mem_usage": True,
                            "use_cache": False
                        }
                    
                    self._local_llm_pipeline = pipeline("text-generation", **pipeline_kwargs)
                    
                    print(f"✅ Local LLM loaded: {model_id} on {device_name}")
                    return True
                    
                except Exception as model_error:
                    print(f"⚠️ Failed to load {model_id}: {model_error}")
                    continue
            
            print("❌ No local LLM models could be loaded")
            return False
            
        except ImportError:
            print("⚠️ Transformers not available. Install with: pip install transformers torch")
            return False
        except Exception as e:
            print(f"⚠️ Local LLM setup failed: {e}")
            return False
    
    def _try_openrouter_client(self) -> bool:
        """Try to initialize OpenRouter client"""
        try:
            import openai
            
            # Check for OpenRouter API key
            api_key = os.getenv('OPENROUTER_API_KEY')
            if not api_key:
                return False
            
            self._llm_client = openai.OpenAI(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1"
            )
            print("🌐 LLM client initialized with OpenRouter API")
            return True
            
        except Exception as e:
            print(f"⚠️ OpenRouter client setup failed: {e}")
            return False
    
    def _try_openai_client(self) -> bool:
        """Try to initialize OpenAI client"""
        try:
            import openai
            
            # Check for OpenAI API key
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                return False
            
            self._llm_client = openai.OpenAI(api_key=api_key)
            print("🤖 LLM client initialized with OpenAI API")
            return True
            
        except Exception as e:
            print(f"⚠️ OpenAI client setup failed: {e}")
            return False
    
    def _get_model_name(self) -> str:
        """Get the appropriate model name based on the provider"""
        # Check if using OpenRouter (has base_url set to openrouter.ai)
        if hasattr(self._llm_client, 'base_url') and 'openrouter.ai' in str(self._llm_client.base_url):
            # OpenRouter format - you can change this to any model available on OpenRouter
            return "openai/gpt-4o-mini"  # Cheaper than GPT-4o
            # Alternative cheaper options:
            # return "meta-llama/llama-3.1-8b-instruct:free"  # Free Llama model
            # return "microsoft/wizardlm-2-8x22b"  # Alternative model
        else:
            # Direct OpenAI
            return "gpt-4o"
    
    def generate_component_descriptions(
        self, 
        component_type: str, 
        context: str = "",
        use_llm: bool = True
    ) -> Dict[str, List[str]]:
        """
        Generate normal and anomaly descriptions for a component type
        
        Args:
            component_type: Type of component (e.g., "metal plate", "circuit board")
            context: Additional context (e.g., "automotive", "electronics")
            use_llm: Whether to use LLM for generation or fallback templates
            
        Returns:
            Dict with 'normal' and 'anomaly' description lists
        """
        if use_llm and self._ensure_llm_client():
            if self._local_llm_pipeline is not None:
                return self._local_llm_generate_descriptions(component_type, context)
            else:
                return self._llm_generate_descriptions(component_type, context)
        else:
            return self._template_generate_descriptions(component_type, context)
    
    def _local_llm_generate_descriptions(self, component_type: str, context: str) -> Dict[str, List[str]]:
        """Generate descriptions using local HuggingFace LLM"""
        try:
            print("🤖 Generating descriptions with local LLM...")
            
            # Create a simpler prompt for local models with shorter output requirement
            prompt = f"""Generate SHORT quality inspection descriptions for {component_type} in {context} context.

Normal condition: List 3 SHORT descriptions (max 10 words each) of perfect {component_type}:
1."""
            
            # Generate with local model (simplified to prevent hanging)
            print(f"🔄 Generating with prompt length: {len(prompt)} chars...")
            
            try:
                # Use very conservative settings to prevent hanging
                generation_kwargs = {
                    "max_new_tokens": 400,  # Very small to prevent timeout
                    "num_return_sequences": 1,
                    "temperature": 0.8,
                    "do_sample": True,
                    "truncation": True,
                    "return_full_text": False,  # Only return generated part
                }
                
                # Add proper tokens if available
                if hasattr(self._local_llm_pipeline.tokenizer, 'eos_token_id') and self._local_llm_pipeline.tokenizer.eos_token_id:
                    generation_kwargs["eos_token_id"] = self._local_llm_pipeline.tokenizer.eos_token_id
                
                if hasattr(self._local_llm_pipeline.tokenizer, 'pad_token_id') and self._local_llm_pipeline.tokenizer.pad_token_id:
                    generation_kwargs["pad_token_id"] = self._local_llm_pipeline.tokenizer.pad_token_id
                
                print(f"🎯 Starting generation with {generation_kwargs['max_new_tokens']} max tokens...")
                
                outputs = self._local_llm_pipeline(prompt, **generation_kwargs)
                
                print(f"✅ Generation completed, got {len(outputs)} outputs")
                
            except Exception as gen_error:
                print(f"⚠️ Generation failed: {gen_error}")
                raise gen_error
            
            generated_text = outputs[0]["generated_text"]
            print(f"📝 Raw generation output: {generated_text[:200]}...")
            
            # Handle both full text and partial text returns
            if isinstance(generated_text, str):
                if len(generated_text) > len(prompt):
                    new_text = generated_text[len(prompt):].strip()
                else:
                    new_text = generated_text.strip()
            else:
                new_text = str(generated_text).strip()
            
            print(f"🎯 Extracted text: {new_text[:100]}...")
            
            # Parse the response (simplified parsing for local models)
            descriptions = self._parse_local_llm_response(new_text, component_type)
            
            print(f"🤖 Local LLM generated {len(descriptions['normal'])} normal and {len(descriptions['anomaly'])} anomaly descriptions")
            return descriptions
            
        except Exception as e:
            print(f"⚠️ Local LLM generation failed: {e}, using template fallback")
            return self._template_generate_descriptions(component_type, context)
    
    def _parse_local_llm_response(self, response_text: str, component_type: str) -> Dict[str, List[str]]:
        """Parse local LLM response into structured descriptions - COMPLETELY REWRITTEN"""
        print(f"🔍 Parsing LLM response: '{response_text[:100]}...'")
        
        normal_descriptions = []
        anomaly_descriptions = []
        
        # Clean up the response text first
        cleaned_text = response_text.strip()
        
        # Remove common prompt artifacts and noise
        noise_patterns = [
            r"Generate.*?descriptions.*?for",
            r"List.*?descriptions.*?of", 
            r"Provide.*?descriptions.*?of",
            r"Describe.*?descriptions.*?of",
            r"Normal condition:",
            r"Defective condition[s]?:",
            r"Abnormal condition:",
            r"Examples?:",
            r"Note:.*",
            r"Thanks.*",
            r"Good luck.*",
            r"Let's.*",
            r"Please.*provide.*",
            r"I'll.*",
            r"Got it.*"
        ]
        
        import re
        for pattern in noise_patterns:
            cleaned_text = re.sub(pattern, "", cleaned_text, flags=re.IGNORECASE)
        
        # Split into sentences/lines and clean each one
        sentences = []
        for line in cleaned_text.split('\n'):
            line = line.strip()
            if line:
                # Remove numbering (1., 2., 3., etc.)
                line = re.sub(r'^\d+\.?\s*', '', line)
                # Remove bullet points (-, *, •)
                line = re.sub(r'^[-*•]\s*', '', line)
                # Remove markdown formatting
                line = re.sub(r'\*\*(.*?)\*\*', r'\1', line)
                line = re.sub(r'\*(.*?)\*', r'\1', line)
                # Clean up extra spaces
                line = re.sub(r'\s+', ' ', line).strip()
                
                if len(line) > 15 and not self._is_noise_line(line):
                    sentences.append(line)
        
        print(f"📝 Cleaned sentences: {sentences}")
        
        # Smart categorization based on content analysis
        for sentence in sentences:
            if self._is_normal_description(sentence):
                normal_descriptions.append(sentence)
                print(f"✅ NORMAL: {sentence[:50]}...")
            elif self._is_anomaly_description(sentence):
                anomaly_descriptions.append(sentence)
                print(f"🚨 ANOMALY: {sentence[:50]}...")
            else:
                print(f"🤷 UNCLEAR: {sentence[:50]}...")
        
        # Ensure we have at least some descriptions with smart templates
        if len(normal_descriptions) < 2:
            normal_descriptions.extend([
                f"pristine {component_type} with smooth uniform surface",
                f"clean {component_type} with consistent color and texture"
            ])
            
        if len(anomaly_descriptions) < 2:
            anomaly_descriptions.extend([
                f"{component_type} with visible surface scratches",
                f"{component_type} showing dents or deformation"
            ])
        
        print(f"🎯 Final parsing: {len(normal_descriptions)} normal, {len(anomaly_descriptions)} anomaly")
        
        result = {
            "normal": normal_descriptions[:5],  # Limit to 5
            "anomaly": anomaly_descriptions[:5]
        }
        
        # Print out the parsed descriptions for debugging
        print("📋 NORMAL descriptions (from local LLM):")
        for i, desc in enumerate(result['normal'], 1):
            print(f"  {i}. {desc}")
        
        print("🚨 ANOMALY descriptions (from local LLM):")
        for i, desc in enumerate(result['anomaly'], 1):
            print(f"  {i}. {desc}")
        
        return result
    
    def _is_noise_line(self, line: str) -> bool:
        """Check if a line is noise/prompt artifact that should be ignored"""
        noise_indicators = [
            "list", "describe", "provide", "generate", "examples", 
            "condition", "requirements", "specifications", "criteria",
            "thanks", "good luck", "let's", "please", "note:",
            "quality inspection", "manufacturing", "plating process"
        ]
        
        line_lower = line.lower()
        
        # Check for noise indicators
        if any(indicator in line_lower for indicator in noise_indicators):
            return True
        
        # Check for incomplete sentences or fragments
        if len(line.split()) < 4:  # Very short sentences are likely fragments
            return True
            
        # Check for lines that are mostly punctuation or special characters
        if len([c for c in line if c.isalpha()]) < len(line) * 0.6:
            return True
            
        return False
    
    def _is_normal_description(self, sentence: str) -> bool:
        """Determine if a sentence describes a normal/good condition"""
        sentence_lower = sentence.lower()
        
        # Strong normal indicators
        normal_keywords = [
            "smooth", "clean", "pristine", "perfect", "flawless", "uniform", 
            "consistent", "undamaged", "free from", "without", "seamless",
            "polished", "shiny", "unblemished", "proper", "no visible",
            "no defects", "no scratches", "no dents", "no damage"
        ]
        
        # Check for strong normal indicators
        normal_score = sum(1 for keyword in normal_keywords if keyword in sentence_lower)
        
        # Negative indicators (things that suggest anomalies)
        anomaly_keywords = [
            "scratch", "dent", "damage", "defect", "crack", "worn", "broken",
            "irregular", "uneven", "rough", "pitted", "corroded", "stained"
        ]
        
        anomaly_score = sum(1 for keyword in anomaly_keywords if keyword in sentence_lower)
        
        # Decide based on keyword balance
        return normal_score > anomaly_score and normal_score > 0
    
    def _is_anomaly_description(self, sentence: str) -> bool:
        """Determine if a sentence describes an anomaly/defect condition"""
        sentence_lower = sentence.lower()
        
        # Strong anomaly indicators
        anomaly_keywords = [
            "scratch", "scratches", "dent", "dents", "damage", "damaged",
            "defect", "defects", "crack", "cracks", "worn", "wear",
            "broken", "irregular", "uneven", "rough", "pitted", "pit",
            "corroded", "corrosion", "stained", "stains", "dirt", "dirty",
            "contamination", "blemish", "imperfect", "faulty"
        ]
        
        # Check for strong anomaly indicators
        anomaly_score = sum(1 for keyword in anomaly_keywords if keyword in sentence_lower)
        
        # Negative indicators (normal condition words)
        normal_keywords = [
            "smooth", "clean", "pristine", "perfect", "flawless", 
            "free from", "without", "no visible", "seamless"
        ]
        
        normal_score = sum(1 for keyword in normal_keywords if keyword in sentence_lower)
        
        # Decide based on keyword balance  
        return anomaly_score > normal_score and anomaly_score > 0
    
    def _llm_generate_descriptions(self, component_type: str, context: str) -> Dict[str, List[str]]:
        """Generate descriptions using LLM"""
        try:
            prompt = f"""
You are an expert in industrial quality inspection. Generate detailed descriptions for:

Component: {component_type}
Context: {context if context else "general industrial setting"}

Generate  descriptions each for:
1. NORMAL/GOOD state - what this component looks like when it's in perfect condition
2. ANOMALY/DEFECT state - what defects, damages, or anomalies might appear

Requirements:
- Make sure the anomaly/ defect descriptions are ONLY about the potential defects mentioned in the context
- Be specific and visual
- Include surface conditions, colors, textures
- Focus on observable characteristics
- Use industrial inspection terminology
- Avoid generic phrases
- Pay attention to the edges of the component, as they are often where defects appear

Format your response as JSON:
{{
    "normal": [
        "description 1",
        "description 2", 
        ...
    ],
    "anomaly": [
        "description 1",
        "description 2",
        ...
    ]
}}
"""
            
            # Use different model names based on provider
            model_name = self._get_model_name()
            
            response = self._llm_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=400
            )
            
            # Parse JSON response
            content = response.choices[0].message.content
            if not content or content.strip() == "":
                raise ValueError("Empty response from LLM")
            
            # Extract JSON from response (handle cases where LLM adds extra text)
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON found in response")
            
            json_str = content[start_idx:end_idx]
            descriptions = json.loads(json_str)
            
            print(f"🧠 Generated {len(descriptions['normal'])} normal and {len(descriptions['anomaly'])} anomaly descriptions")
            
            # Print out the generated descriptions for debugging
            print("📋 NORMAL descriptions:")
            for i, desc in enumerate(descriptions['normal'], 1):
                print(f"  {i}. {desc}")
            
            print("🚨 ANOMALY descriptions:")
            for i, desc in enumerate(descriptions['anomaly'], 1):
                print(f"  {i}. {desc}")
            
            return descriptions
            
        except Exception as e:
            print(f"⚠️ LLM generation failed: {e}, using template fallback")
            return self._template_generate_descriptions(component_type, context)
    
    def _template_generate_descriptions(self, component_type: str, context: str) -> Dict[str, List[str]]:
        """Fallback template-based description generation"""
        # Template descriptions based on common industrial scenarios
        templates = {
            "normal": [
                f"pristine {component_type} with smooth uniform surface",
                f"clean {component_type} with consistent color and texture",
                f"undamaged {component_type} with proper geometric shape",
                f"flawless {component_type} surface without any marks",
                f"perfect condition {component_type} with no visible defects"
            ],
            "anomaly": [
                f"{component_type} with visible scratches on surface",
                f"{component_type} showing signs of corrosion or rust",
                f"{component_type} with dents or deformation",
                f"{component_type} displaying discoloration or stains",
                f"{component_type} with cracks or structural damage",
                f"{component_type} showing wear patterns or erosion",
                # f"{component_type} with contamination or foreign particles"
            ]
        }
        
        # Customize based on context
        if "metal" in component_type.lower() or "steel" in component_type.lower():
            templates["anomaly"].extend([
                f"{component_type} with oxidation spots",
                f"{component_type} showing metal fatigue signs"
            ])
        
        if "electronic" in context.lower() or "circuit" in component_type.lower():
            templates["anomaly"].extend([
                f"{component_type} with burned components",
                f"{component_type} showing electrical damage"
            ])
        
        print(f"🏭 Generated template descriptions for {component_type}")
        
        # Print out the template descriptions for debugging
        print("📋 NORMAL descriptions (template):")
        for i, desc in enumerate(templates['normal'], 1):
            print(f"  {i}. {desc}")
        
        print("🚨 ANOMALY descriptions (template):")
        for i, desc in enumerate(templates['anomaly'], 1):
            print(f"  {i}. {desc}")
        
        return templates
    
    def detect_anomalies_with_llm_clip(
        self,
        image_path: str,
        component_type: str,
        context: str = "",
        confidence_threshold: float = 0.3,
        similarity_threshold: float = 0.7,
        use_llm: bool = True
    ) -> Dict:
        """
        CORRECTED anomaly detection pipeline using LLM + GroundingDINO + CLIP
        
        Workflow:
        1. LLM generates detailed normal/anomaly descriptions for the component
        2. GroundingDINO finds component instances using SIMPLE user prompt (not LLM descriptions)
        3. CLIP analyzes each detected region against LLM descriptions for anomaly detection
        
        Args:
            image_path: Path to input image
            component_type: Simple component name for GroundingDINO (e.g., "metal plate")
            context: Additional context for LLM description generation
            confidence_threshold: GroundingDINO confidence threshold
            similarity_threshold: CLIP similarity threshold for anomaly detection
            use_llm: Whether to use LLM for description generation
            
        Returns:
            Detection results with anomaly classification
        """
        try:
            print(f"🔬 Starting LLM+CLIP anomaly detection for: {os.path.basename(image_path)}")
            print(f"📋 Component: {component_type}, Context: {context}")
            
            # Step 1: Generate detailed descriptions for CLIP analysis (NOT for GroundingDINO)
            print("🧠 Step 1: Generating detailed normal/anomaly descriptions for CLIP...")
            descriptions = self.generate_component_descriptions(component_type, context, use_llm)
            
            # Step 2: Use GroundingDINO with SIMPLE user prompt to find components
            print("🎯 Step 2: Localizing components with GroundingDINO using simple prompt...")
            # Use ONLY the simple component type for GroundingDINO - NOT the verbose LLM descriptions
            grounding_prompts = [component_type]  # Simple, clear object name only
            
            print(f"🔍 GroundingDINO searching for: '{component_type}' (simple prompt)")
            
            grounding_result = self.grounding_dino.annotate_with_prompts(
                image_path, grounding_prompts, confidence_threshold
            )
            
            if grounding_result['status'] != 'success' or not grounding_result['detections']:
                return {
                    "status": "success",
                    "message": f"No {component_type} detected in image",
                    "detections": [],
                    "component_type": component_type,
                    "descriptions_used": descriptions,
                    "annotated_image_path": None,
                    "image_base64": None
                }
            
            detections = grounding_result['detections']
            print(f"✅ Found {len(detections)} component instances")
            
            # Filter to select the best detection (central + reasonably sized)
            if len(detections) > 1:
                print("🎯 Multiple detections found, selecting best one...")
                detections = [self._select_best_detection(detections, image_path)]
                print(f"✅ Selected best detection (central + adequately sized)")
            
            # Step 3: CLIP-based anomaly detection using LLM descriptions on detected regions
            print("🔍 Step 3: CLIP analyzing detected regions against LLM descriptions...")
            if not self._ensure_clip_loaded():
                return {
                    "status": "error",
                    "message": "CLIP model not available"
                }
            
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Analyze each detected region
            anomaly_results = []
            for i, detection in enumerate(detections):
                print(f"🔬 Analyzing region {i+1}: {detection['class']}")
                
                # Extract region of interest
                bbox = detection['bbox']
                x1, y1, x2, y2 = [int(coord) for coord in bbox]
                roi = image[y1:y2, x1:x2]
                
                if roi.size == 0:
                    continue
                
                # CLIP analysis
                anomaly_score, anomaly_type, similarity_scores = self._analyze_roi_with_clip(
                    roi, descriptions, similarity_threshold
                )
                
                anomaly_result = {
                    "region_index": i,
                    "grounding_class": detection['class'],
                    "grounding_confidence": detection['confidence'],
                    "bbox": bbox,
                    "anomaly_score": anomaly_score,
                    "is_anomaly": anomaly_score < similarity_threshold,
                    "anomaly_type": anomaly_type,
                    "similarity_scores": similarity_scores,
                    "severity": self._assess_anomaly_severity(anomaly_score, similarity_threshold)
                }
                
                anomaly_results.append(anomaly_result)
            
            # Step 4: Create annotated visualization
            print("🎨 Step 4: Creating annotated visualization...")
            annotated_image_path, image_base64 = self._create_llm_clip_visualization(
                image_path, anomaly_results, descriptions, component_type
            )
            
            # Summary statistics
            total_regions = len(anomaly_results)
            anomaly_regions = len([r for r in anomaly_results if r['is_anomaly']])
            
            result = {
                "status": "success",
                "component_type": component_type,
                "context": context,
                "total_regions_detected": total_regions,
                "anomaly_regions_detected": anomaly_regions,
                "anomaly_rate": anomaly_regions / total_regions if total_regions > 0 else 0.0,
                "detections": anomaly_results,
                "descriptions_used": descriptions,
                "annotated_image_path": annotated_image_path,
                "image_base64": image_base64,
                "confidence_threshold": confidence_threshold,
                "similarity_threshold": similarity_threshold,
                "pipeline_type": "llm_clip_anomaly_detection"
            }
            
            print(f"✅ LLM+CLIP analysis completed: {anomaly_regions}/{total_regions} anomalies detected")
            return result
            
        except Exception as e:
            print(f"❌ LLM+CLIP anomaly detection failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                "status": "error",
                "message": f"Analysis failed: {str(e)}"
            }
    
    def _analyze_roi_with_clip(
        self, 
        roi_image: np.ndarray, 
        descriptions: Dict[str, List[str]], 
        similarity_threshold: float
    ) -> Tuple[float, str, Dict]:
        """
        Analyze ROI using CLIP for anomaly detection
        
        Returns:
            - anomaly_score: Similarity score (higher = more normal)
            - anomaly_type: Best matching anomaly description
            - similarity_scores: All similarity scores
        """
        try:
            import torch
            
            # Convert BGR to RGB
            roi_rgb = cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB)
            
            # Prepare all text descriptions and truncate for CLIP (max 77 tokens)
            normal_texts = descriptions["normal"]
            anomaly_texts = descriptions["anomaly"]
            
            # Truncate texts to fit CLIP's token limit (77 tokens ≈ 50-60 words)
            def truncate_text(text, max_words=50):
                words = text.split()
                if len(words) > max_words:
                    return ' '.join(words[:max_words])
                return text
            
            normal_texts = [truncate_text(text) for text in normal_texts]
            anomaly_texts = [truncate_text(text) for text in anomaly_texts]
            all_texts = normal_texts + anomaly_texts
            
            print(f"🔤 Prepared {len(all_texts)} text descriptions (truncated for CLIP)")
            
            # CLIP processing with error handling for text length
            try:
                inputs = self._clip_processor(
                    text=all_texts, 
                    images=roi_rgb, 
                    return_tensors="pt", 
                    padding=True,
                    truncation=True,  # Enable truncation in the processor
                    max_length=77     # Explicitly set max length
                )
            except Exception as clip_error:
                print(f"⚠️ CLIP processing failed, using shorter descriptions: {clip_error}")
                # Fallback to very short descriptions
                short_normal = ["normal " + descriptions["normal"][0].split()[0] if descriptions["normal"] else "normal component"]
                short_anomaly = ["defective " + descriptions["anomaly"][0].split()[0] if descriptions["anomaly"] else "defective component"]
                all_texts = short_normal + short_anomaly
                
                inputs = self._clip_processor(
                    text=all_texts, 
                    images=roi_rgb, 
                    return_tensors="pt", 
                    padding=True,
                    truncation=True,
                    max_length=77
                )
            
            if torch.cuda.is_available() and self._clip_model.training == False:
                inputs = {k: v.cuda() if hasattr(v, 'cuda') else v for k, v in inputs.items()}
            
            # Get embeddings
            with torch.no_grad():
                outputs = self._clip_model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = torch.softmax(logits_per_image, dim=-1)
            
            # Convert to numpy
            similarities = probs.cpu().numpy().flatten()
            
            # Calculate scores
            normal_similarities = similarities[:len(normal_texts)]
            anomaly_similarities = similarities[len(normal_texts):]
            
            # Anomaly score = best normal similarity
            normal_score = float(np.max(normal_similarities))
            best_anomaly_score = float(np.max(anomaly_similarities))
            
            # Find best matching anomaly type
            best_anomaly_idx = np.argmax(anomaly_similarities)
            anomaly_type = anomaly_texts[best_anomaly_idx]
            
            # Compile similarity scores
            similarity_scores = {
                "normal_max": normal_score,
                "normal_mean": float(np.mean(normal_similarities)),
                "anomaly_max": best_anomaly_score,
                "anomaly_mean": float(np.mean(anomaly_similarities)),
                "best_anomaly_type": anomaly_type,
                "all_normal_scores": normal_similarities.tolist(),
                "all_anomaly_scores": anomaly_similarities.tolist()
            }
            
            return normal_score, anomaly_type, similarity_scores
            
        except Exception as e:
            print(f"⚠️ CLIP analysis error: {e}")
            return 0.5, "unknown_anomaly", {}
    
    def _assess_anomaly_severity(self, anomaly_score: float, threshold: float) -> str:
        """Assess the severity of detected anomaly"""
        if anomaly_score >= threshold:
            return "normal"
        
        severity_ratio = anomaly_score / threshold
        
        if severity_ratio < 0.5:
            return "high"
        elif severity_ratio < 0.8:
            return "medium"
        else:
            return "low"
    
    def _create_llm_clip_visualization(
        self,
        image_path: str,
        anomaly_results: List[Dict],
        descriptions: Dict[str, List[str]],
        component_type: str
    ) -> Tuple[str, str]:
        """Create annotated visualization showing anomaly detection results"""
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        annotated_img = image.copy()
        
        # Color scheme
        colors = {
            'normal': (0, 255, 0),      # Green
            'low': (0, 255, 255),       # Yellow
            'medium': (0, 165, 255),    # Orange
            'high': (0, 0, 255),        # Red
        }
        
        # Draw each detected region
        for result in anomaly_results:
            bbox = result['bbox']
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            
            severity = result['severity']
            color = colors.get(severity, (128, 128, 128))
            
            # Draw bounding box
            thickness = 4 if result['is_anomaly'] else 2
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, thickness)
            
            # Prepare label
            if result['is_anomaly']:
                label = f"ANOMALY: {result['anomaly_type'][:20]}..."
                score_text = f"Score: {result['anomaly_score']:.3f}"
            else:
                label = f"NORMAL: {result['grounding_class']}"
                score_text = f"Score: {result['anomaly_score']:.3f}"
            
            # Draw label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            score_size = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            
            max_width = max(label_size[0], score_size[0])
            total_height = label_size[1] + score_size[1] + 20
            
            # Background rectangle
            bg_y1 = y1 - total_height - 10
            bg_y2 = y1 - 5
            
            if bg_y1 < 0:  # If too close to top, draw below
                bg_y1 = y2 + 5
                bg_y2 = y2 + total_height + 10
            
            cv2.rectangle(annotated_img, (x1, bg_y1), (x1 + max_width + 15, bg_y2), color, -1)
            
            # Draw text
            text_y = bg_y1 + label_size[1] + 5
            cv2.putText(annotated_img, label, (x1 + 5, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            text_y += score_size[1] + 5
            cv2.putText(annotated_img, score_text, (x1 + 5, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add title and summary
        title = f"LLM+CLIP Anomaly Detection: {component_type}"
        anomaly_count = len([r for r in anomaly_results if r['is_anomaly']])
        summary = f"Anomalies: {anomaly_count}/{len(anomaly_results)} regions"
        
        # Title background
        cv2.rectangle(annotated_img, (10, 10), (10 + len(title) * 12, 60), (0, 0, 0), -1)
        cv2.putText(annotated_img, title, (15, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated_img, summary, (15, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Save result
        output_filename = f"llm_clip_anomaly_{Path(image_path).stem}_{uuid.uuid4().hex[:8]}.jpg"
        output_path = os.path.join(self.results_dir, output_filename)
        cv2.imwrite(output_path, annotated_img)
        
        # Convert to base64
        _, buffer = cv2.imencode('.jpg', annotated_img)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return output_path, image_base64
    
    def batch_detect_anomalies(
        self,
        image_paths: List[str],
        component_type: str,
        context: str = "",
        confidence_threshold: float = 0.3,
        similarity_threshold: float = 0.7,
        use_llm: bool = True
    ) -> Dict:
        """Batch processing for multiple images"""
        results = []
        
        # Generate descriptions once for the batch
        descriptions = self.generate_component_descriptions(component_type, context, use_llm)
        
        for image_path in image_paths:
            print(f"📸 Processing {os.path.basename(image_path)}...")
            result = self.detect_anomalies_with_llm_clip(
                image_path, component_type, context, 
                confidence_threshold, similarity_threshold, use_llm
            )
            result['filename'] = os.path.basename(image_path)
            results.append(result)
        
        # Generate batch summary
        successful_results = [r for r in results if r['status'] == 'success']
        total_regions = sum(r.get('total_regions_detected', 0) for r in successful_results)
        total_anomalies = sum(r.get('anomaly_regions_detected', 0) for r in successful_results)
        
        summary = {
            "total_images": len(image_paths),
            "successful_analyses": len(successful_results),
            "failed_analyses": len(results) - len(successful_results),
            "total_regions_analyzed": total_regions,
            "total_anomalies_detected": total_anomalies,
            "overall_anomaly_rate": total_anomalies / total_regions if total_regions > 0 else 0.0,
            "component_type": component_type,
            "context": context
        }
        
        return {
            "status": "success",
            "results": results,
            "summary": summary,
            "descriptions_used": descriptions,
            "pipeline_type": "llm_clip_batch_anomaly_detection"
        }


# Create global instance
llm_clip_anomaly_service = LLMCLIPAnomalyService()