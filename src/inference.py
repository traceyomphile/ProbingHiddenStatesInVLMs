# src.inference.py
import math
import re
import torch
import gzip
import pickle
import numpy as np
from pathlib import Path
from PIL import Image
from dataclasses import dataclass, replace
from transformers import (
    AutoProcessor,
    PreTrainedModel, 
    ProcessorMixin,
)

# Keep code reusable across different versions
try:
    from transformers import AutoModelForMultimodalLM as _AutoModel
except ImportError:
    try:
        from transformers import AutoModelForImageTextToText as _AutoModel
    except ImportError:
        from transformers import AutoModelForVision2Seq as _AutoModel

_MODEL_NAME = 'HuggingFaceTB/SmolVLM-256M-Instruct'

@dataclass
class InferenceResult:
    image_id: int
    category: str
    question_type: str
    ground_truth: bool
    generated_text: str
    parsed_answer: bool | None      # None means "unclear"
    confidence: float               # Pronbability anssigned to the generated answer token(s)
    hidden_states: dict[int, np.ndarray]    # layer index -> (seq_len, hidden_dim) array --- raw, unpooled

def _model_device(model) -> torch.device:
    """Find the device on which the model parameters live."""
    return next(model.parameters()).device

def _built_prompt(processor, question: str) -> str:
    """
    Build the multimodal chat prompt.
    We explicitly request only yes/no so parsing is deterministic
    and generated answers remain short.
    """
    messages = [
        {
            'role': 'user',
            'content': [
                {'type': 'image'},
                {
                    'type': 'text',
                    'text': question,
                },
            ],
        }
    ]

    return processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

def _prepare_inputs(model, processor, image, question: str) -> dict:
    """Convert the imge and question into tensors ready for SmolVLM."""
    prompt = _built_prompt(processor, question)
    inputs = processor(
        text=prompt,
        images=[image],
        return_tensors='pt',
    )

    device = _model_device(model)

    return {
        key: value.to(device) if torch.is_tensor(value) else value 
        for key, value in inputs.items()
    }

def _parse_answer(text: str) -> bool | None:
    """
    Convert a generated answer into:
        True -> yes
        False -> no
        None -> unclear
    """
    cleaned = text.strip().lower()

    # Occasionally decodeed text can contain an 'assistant: prefix.
    cleaned = re.sub(r'^assistant\s*:\s*', '', cleaned)

    match = re.fullmatch(
        r"""[\s"'`]*(yes|no)[\s"'`.!?,;:]*""",
        cleaned,
    )

    if match is None:
        return None

    return match.group(1) == 'yes'

def _extract_image_hidden_states(model, inputs: dict):
    """
    @author ChatGPT
    Run the vision encoder omce and return the projected image features.
    SmolVLM is based on Idefics3. Idefics3 allows these image hidden states to be passed
    back to the model instead of pixel_values, avoiding another vision-encoder pass.
    """
    if not hasattr(model, 'get_image_features'):
        return None

    if 'pixel_values' not in inputs:
        return None

    kwargs = {
        'pixel_values': inputs['pixel_values'],
        'return_dict': True,
    }

    if 'pixel_attention_mask' in inputs:
        kwargs['pixel_attention_mask'] = inputs['pixel_attention_mask']

    with torch.inference_mode():
        image_outputs = model.get_image_features(**kwargs)

    # For idefics3/SmolVLM, pooler_output contains the vision features after the modality projection
    image_hiddens_states = getattr(image_outputs, 'pooler_output', None)

    if image_hiddens_states is None:
        return None

    return image_hiddens_states.detach()

def _generation_inputs(inputs: dict, image_hidden_states=None) -> dict:
    """
    Construct model inputs either using cached image features or original pixels.
    """
    if image_hidden_states is None:
        return dict(inputs)

    # Idefics3 forbids supplying both pixel_values and image_hidden_states simultaneously
    result = {
        key: value 
        for key, value in inputs.items()
        if key not in {'pixel_values', 'pixel_attention_mask'}
    }

    result['image_hidden_states'] = image_hidden_states

    return result

def _answer_confidence(generation_output, generated_tokens_ids, processor) -> float:
    """
    Calculate the probability of the generated answer sequence.
    """
    tokenizer = getattr(processor, 'tokenizer', None)

    special_ids = set()

    if tokenizer is not None:
        special_ids = set(tokenizer.all_special_ids)

    log_probabilites: list[float] = []

    for scores, token_id_tensor in zip(generation_output.scores, generated_tokens_ids):
        token_id = int(token_id_tensor.item())

        if token_id in special_ids:
            continue

        log_probs = torch.log_softmax(
            scores[0].float(),
            dim=-1,
        )

        token_log_prob = float(log_probs[token_id].item())
        log_probabilites.append(token_log_prob)

    if not log_probabilites:
        return 0.0

    return float(math.exp(sum(log_probabilites)))

def _convert_hidden_states(model, hidden_states) -> dict[int, np.ndarray]:
    """
    Convert transformer-layer hidden states to NumPy arrays.
    Exclude the embedding output layer.
    """
    if hidden_states is None:
        raise RuntimeError("Model did not return hidden states despite output_hidden_states=True")

    text_config = getattr(model.config, 'text_config', None)

    if text_config is not None:
        num_layers = getattr(
            text_config,
            'num_hidden_layers',
            len(hidden_states) - 1,
        )
    else:
        num_layers = getattr(
            model.config,
            'num_hidden_layers',
            len(hidden_states) - 1,
        )

    transformer_states = hidden_states[-num_layers:]

    result: dict[int, np.ndarray] = {}

    for layer_index, state in enumerate(transformer_states):
        # Remove the batch dimension since the batch size is 1.
        state = state[0].detach().cpu()

        # NumPy does not support bfloat16 directly
        if state.dtype == torch.bfloat16:
            state = state.float()

        result[layer_index] = state.numpy()

    return result

def _run_single_example_implementation(model, processor, image, question: str, image_hidden_states=None) -> InferenceResult:
    """
    Allows run_inference_on_manifest() to pass a cached vision-encoder output.
    """
    inputs = _prepare_inputs(model, processor, image, question)

    # Compute image representation if not supplied
    if image_hidden_states is None:
        image_hidden_states = _extract_image_hidden_states(model, inputs)

    model_inputs = _generation_inputs(inputs, image_hidden_states)

    prompt_length = model_inputs['input_ids'].shape[-1]

    # Generate the yes/no answer and keep generation probs
    with torch.inference_mode():
        generation_output = model.generate(
            **model_inputs,
            max_new_tokens=4,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )

    full_sequence = generation_output.sequences[0]

    generated_token_ids = full_sequence[prompt_length:]

    generated_text = processor.decode(
        generated_token_ids,
        skip_special_tokens=True,
    ).strip()

    parsed_answer = _parse_answer(generated_text)

    confidence = _answer_confidence(
        generation_output,
        generated_token_ids,
        processor,
    )

    # Obtain one complete sequence of hidden vectors at every transformer layer.
    full_input_ids = full_sequence.unsqueeze(0)

    forward_inputs = {
        'input_ids': full_input_ids,
        'attention_mask': torch.ones_like(full_input_ids),
    }

    if image_hidden_states is not None:
        forward_inputs['image_hidden_states'] = image_hidden_states
    else:
        # Fallback if cannot cache the vision representation.
        if 'pixel_values' in inputs:
            forward_inputs['pixel_values'] = inputs['pixel_values']

        if 'pixel_attention_mask' in inputs:
            forward_inputs['pixel_attention_mask'] = inputs['pixel_attention_mask']

    with torch.inference_mode():
        forward_output = model(
            **forward_inputs,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

    hidden_states = _convert_hidden_states(model, forward_output.hidden_states)

    return InferenceResult(
        image_id=-1,
        category="",
        question_type="",
        ground_truth=False,
        generated_text=generated_text,
        parsed_answer=parsed_answer,
        confidence=confidence,
        hidden_states=hidden_states,
    )

def _coerce_ground_truth(value) -> bool:
    """
    Convert ground_truth values loaded from CSV/dictionaries to bool.
    """
    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    if isinstance(value, str):
        cleaned = value.strip().lower()

        if cleaned == 'true': return True
        if cleaned == 'false': return False

    raise ValueError(f'Cannot interpret ground_truth value as boolean {value!r}')

def _image_path(image_dir: str, image_id: int, row: dict) -> Path:
    """
    Resolve the COCO image filename.
    """
    root = Path(image_dir)

    file_name = row.get('file_name')

    if file_name:
        candidate = root / str(file_name)

        if candidate.exists(): return candidate

    candidate = root / f'{image_id:012d}.jpg'

    if candidate.exists(): return candidate

    raise FileNotFoundError(
        f'Could not find image for COOC image_id={image_id} inside {root}'
    )

def load_model(model_name: str, device: str) -> tuple[PreTrainedModel, ProcessorMixin]:
    if device.startswith('cuda') and not torch.cuda.is_available():
        device = 'cpu'      # Fallback

    processor: ProcessorMixin = AutoProcessor.from_pretrained(model_name)

    model: PreTrainedModel = _AutoModel.from_pretrained(
        model_name, 
        torch_dtype=torch.bfloat16,
        attn_implementation='eager',
    ).to(device)

    model.eval()

    # Explicitly freeze the model.
    for p in model.parameters():
        p.requires_grad_(False)

    return model, processor

def run_single_example(model, processor, image, question: str) -> InferenceResult:
    """
    One (image, question) pair in, one InferenceResult out. Must not write to disk -
    that's the caller's job, so this function stays testable on a single toy example.
    """
    return _run_single_example_implementation(model, processor, image, question)

def run_inference_on_manifest(model, processor, manifest: list[dict], image_dir: str) -> list[InferenceResult]:
    """
    Iterates the manifest, caching the vision-encoder pass per image_id rather than
    per question if your architecture allows it. Returns one InferenceResult per row.
    """
    if not manifest: 
        return []

    # Group rows by image to run the vision encoder once
    grouped_rows: dict[int, list[tuple[int, dict]]] = {}

    for index, row in enumerate(manifest):
        image_id = int(row['image_id'])

        grouped_rows.setdefault(image_id, []).append(
            (index, row)
        )

    ordered_results: list[InferenceResult | None] = [
        None
        for _ in manifest
    ]

    for image_id, image_rows in grouped_rows.items():
        first_row = image_rows[0][1]

        path = _image_path(image_dir, image_id, first_row)

        with Image.open(path) as opened_image:
            image = opened_image.convert('RGB')

        # cache vision output ONCE for this image
        image_hidden_states = None
        if hasattr(model, 'get_image_features'):
            first_question = str(first_row['question'])

            first_inputs = _prepare_inputs(
                model,
                processor,
                image,
                first_question,
            )

            image_hidden_states = _extract_image_hidden_states(model, first_inputs)

        for original_index, row in image_rows:
            result = _run_single_example_implementation(
                model,
                processor,
                image,
                str(row['question']),
                image_hidden_states=image_hidden_states,
            )

            # Add metadata that run_single_example cannot know
            result = replace(
                result,
                image_id=image_id,
                category=str(row['category']),
                question_type=str(row['question_type']),
                ground_truth=_coerce_ground_truth(row['ground_truth']),
            )

            ordered_results[original_index] = result

        # The cached feature tensor is no longer needed after the three questions for this image
        del image_hidden_states

    if any(result is None for result in ordered_results):
        raise RuntimeError(
            'Inference failed to produce one result per manifest row.'
        )

    return [
        result
        for result in ordered_results
        if result is not None
    ]

def save_results(results: list[InferenceResult], path: str) -> None:
    """
    Save inference results, including the unpooled NumPy hidden-state arrays.

    gzip + pickle is used because the hidden states are large multidimensional
    NumPy arrays and are unsuitable for normal JSON serialization.
    """
    output_path = Path(path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    serialisable = []

    for result in results:
        serialisable.append(
            {
                'image_id': result.image_id,
                'category': result.category,
                'question_type': result.question_type,
                'ground_truth': result.ground_truth,
                'generated_text': result.generated_text,
                'parsed_answer': result.parsed_answer,
                'confidence': result.confidence,
                'hidden_states': result.hidden_states,
            }
        )

    with gzip.open(output_path, 'wb') as file:
        pickle.dump(
            serialisable,
            file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

def load_results(path: str) -> list[InferenceResult]:
    """
    Load results previously written by save_results().
    """
    input_path = Path(path)

    with gzip.open(input_path, 'rb') as file:
        serialised = pickle.load(file)

    results: list[InferenceResult] = []

    for item in serialised:
        hidden_states = {
            int(layer): np.asarray(values)
            for layer, values in item['hidden_states'].items()
        }

        results.append(
            InferenceResult(
                image_id=int(item['image_id']),
                category=str(item['category']),
                question_type=str(item['question_type']),
                ground_truth=bool(item['ground_truth']),
                generated_text=str(item['generated_text']),
                parsed_answer=item['parsed_answer'],
                confidence=float(item['confidence']),
                hidden_states=hidden_states,
            )
        )

    return results

