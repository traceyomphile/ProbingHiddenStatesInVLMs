# src.inference.py
import math
import os
import hashlib
import json
import re
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from dataclasses import dataclass, replace
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import (
    AutoProcessor,
    PreTrainedModel, 
    ProcessorMixin,
    AutoModelForImageTextToText as _AutoModel,
)

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

"""
Helper Methods
"""
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

def _convert_generation_hidden_states(model, generation_hidden_states) -> dict[int, np.ndarray]:
    """
    Convert generation hidden states int:
        layer_index -> (seq_len, hidden_dim)

    The embedding output is excluded.

    BF16 is converted to FP16 before NumPy conversion so that storage
    remains 2 bytes per value rather than being doubled to FP32.
    """
    if generation_hidden_states is None:
        raise RuntimeError("generate() did not return hidden states despite output_hidden_states=True")

    text_config = getattr(model.config, 'text_config', None)

    if text_config is not None:
        num_layers = text_config.num_hidden_layers
    else:
        num_layers = model.config.num_hidden_layers

    result: dict[int, np.ndarray] = {}

    for layer_index in range(num_layers):
        layer_parts: list[torch.Tensor] = []

        for step_hidden_states in generation_hidden_states:
            # Remove embedding state by taking the final num_layers entries
            transformer_states = step_hidden_states[-num_layers:]

            state = transformer_states[layer_index]

            # Remove the batch dimension since the batch size is 1.
            state = state[0].detach()

            # NumPy does not support bfloat16 directly
            if state.dtype == torch.bfloat16:
                state = state.to(torch.float16)     # Keep it at 2 bytes per value

            state = state.cpu()
            layer_parts.append(state)

        full_layer_sequence = torch.cat(
            layer_parts,
            dim=0
        )
        result[layer_index] = full_layer_sequence.numpy()

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
            output_hidden_states=True
        )

    hidden_states = _convert_generation_hidden_states(model, generation_output.hidden_states)
    
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

def _result_metadata(result: InferenceResult, layers: list[int]) -> dict:
    """
    Convert non-tensor result fields to JSON-serialisable metadata.
    """
    return {
        "image_id": result.image_id,
        "category": result.category,
        "question_type": result.question_type,
        "ground_truth": result.ground_truth,
        "generated_text": result.generated_text,
        "parsed_answer": result.parsed_answer,
        "confidence": result.confidence,
        "layers": layers,
    }

def _build_safeternsors_payload(
        results: list[InferenceResult], 
        extra_metadata: dict[str, str] | None = None
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """
    Convert InferenceResults into a SafeTensors tensor dictionary plus string metadata.
    """
    tensors: dict[str, torch.Tensor] = {}
    metadata: dict[str, str] = {
        'format': 'vlm_inference_results_v1',
        'result_count': str(len(results))
    }

    for result_index, result in enumerate(results):
        layers = sorted(results.hidden_states.keys())

        for layer_index in layers:
            array = np.asarray(result.hidden_states[layer_index])

            # SafeTensors expects dense contiguous tensors.
            array = np.ascontiguousarray(array)

            tensor = torch.from_numpy(array).contiguous()

            key = (
                f"result_{result_index}."
                f"layer_{layer_index}"
            )

            tensors[key] = tensor

        metadata[f"result_{result_index}"] = json.dumps(
            _result_metadata(result, layers),
            separators=(",", ":")
        )

    if extra_metadata:
        for key, value in extra_metadata.items():
            metadata[str(key)] = str(value)

    return tensors, metadata

def _atomic_save_results(results: list[InferenceResult], path: Path, extra_metadata: dict[str, str] | None = None) -> None:
    """
    Automatically save results using SafeTensors.
    Data is first written to an temp file in the same dir, then os.replace() moves it into place.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    tensors, metadata = _build_safeternsors_payload(results, extra_metadata)

    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")

    try:
        save_file(tensors, str(temp_path))

        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()

def _load_results_and_metadata(path: Path) -> tuple[list[InferenceResult], dict[str, str]]:
    """
    Load a SafeTensors inference-results file.
    """
    results: list[InferenceResult] = []

    with safe_open(str(path), framework='pt', device='cpu') as file:
        metadata = file.metadata() or {}
        result_count = int(metadata.get('result_count', '0'))

        for result_index in range(result_count):
            metadata_key = f"result_{result_index}"

            if metadata_key not in metadata:
                raise RuntimeError(f"Missing metadata entry {metadata_key!r} in {path}")

            item = json.loads(metadata[metadata_key])
            hidden_states: dict[int, np.ndarray] = {}

            for layer_index in item['layers']:
                tensor_key = (
                    f"result_{result_index}."
                    f"layer_{layer_index}"
                )

                tensor = file.get_tensor(tensor_key)

                # Make an independent NumPy copy before the SafeTensors file handle is closed.
                hidden_states[int(layer_index)] = tensor.cpu().numpy().copy()

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

    return results, metadata

def _checkpoint_path(checkpoint_dir: Path, manifest_index: int, image_id: int) -> Path:
    """One checkpoint file per QA pair."""
    return (
        checkpoint_dir /
        (
            f"row_{manifest_index:06d}_"
            f"image_{image_id:012d}"
            f".safetensors"
        )
    )

def _load_checkpoint(path: Path) -> InferenceResult | None:
    """Load a valid checkpoint"""
    if not path.exists(): 
        return None

    try:
        results, metadata = _load_results_and_metadata(path)

        if len(results) != 1:
            return None

        return results[0]
    except Exception:
        # A bad checkpoint should not destroy the run.
        return None

def _save_checkpoint(result: InferenceResult, path: Path, manifest_index: int) -> None:
    """
    Atomically persist one completed QA result.
    """ 
    _atomic_save_results(
        [result], 
        path, 
        extra_metadata={
            'manifest_index': str(manifest_index)
        },
    )

def _run_manifest(model, processor, manifest: list[dict], image_dir: str, checkpoint_dir: Path | None) -> list[InferenceResult]:
    """Shared implementation for normal and resumable inference."""
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

    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for image_id, image_rows in grouped_rows.items():
        # A resumed run need not load the image when all three questions already have valid checkpoint
        missing_rows: list[tuple[int, dict]] = []

        for original_index, row in image_rows:
            if checkpoint_dir is None:
                missing_rows.append((original_index, row))
                continue

            checkpoint = _checkpoint_path(checkpoint_dir, original_index, image_id)
            cached_result = _load_checkpoint(checkpoint)

            if cached_result is not None:
                ordered_results[original_index] = cached_result
                continue

            missing_rows.append((original_index, row))

        # Every question belonging to this image is already done.
        if not missing_rows:
            continue

        # Load the image only if at least one question still needs work.
        first_missing_row = missing_rows[0][1]

        path = _image_path(image_dir, image_id, first_missing_row)

        with Image.open(path) as opened_image:
            image = opened_image.convert('RGB')

        # cache vision output ONCE for this image
        image_hidden_states = None
        if hasattr(model, 'get_image_features'):
            first_question = str(first_missing_row['question'])

            first_inputs = _prepare_inputs(
                model,
                processor,
                image,
                first_question,
            )

            image_hidden_states = _extract_image_hidden_states(model, first_inputs)

            # first_inputs is no longer required
            del first_inputs

        # Run unfinished questions.
        for original_index, row in missing_rows:
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

            # Once this call returns successfully, this QA pair survives a later crash
            if checkpoint_dir is not None:
                checkpoint = _checkpoint_path(checkpoint_dir, original_index, image_id)
                _save_checkpoint(result, checkpoint, original_index)

        # The cached feature tensor is no longer needed after the three questions for this image
        del image_hidden_states

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if any(result is None for result in ordered_results):
        raise RuntimeError(
            'Inference failed to produce one result per manifest row.'
        )

    return [
        result
        for result in ordered_results
        if result is not None
    ]

# Additional resumable interface
def run_inference_resumable(model, processor, manifest: list[dict], image_dir: str, checkpoint_dir: str) -> list[InferenceResult]:
    """
    Run inference with an atomic SafeTensors checkpoint after every QA pair.

    Existing valid checkpoints are automatically loaded and skipped.
    """
    return _run_manifest(model, processor, manifest, image_dir, checkpoint_dir=Path(checkpoint_dir))

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
    return _run_manifest(model, processor, manifest, image_dir, checkpoint_dir=None)

def save_results(results: list[InferenceResult], path: str) -> None:
    """
    Save all inference results as SafeTensors.

    Hidden states are stored as tensors.
    Results metadata is stored in the SafeTensors metadata header.
    """
    _atomic_save_results(results, Path(path))

def load_results(path: str) -> list[InferenceResult]:
    """
    Load results previously written by save_results().
    """
    results, _ = _load_results_and_metadata(Path(path))

    return results

