# src.inference.py
import os
import json
import re
import torch
from tqdm import tqdm
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

RESULT_FORMAT = 'vlm_inference_results_v1'
MAX_BATCH_BYTES = 8 * 1024 * 1024  # 8MB
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

    def __repr__(self):
        return (
            f'image_id: {self.image_id}\n'
            f'category: {self.category}\n'
            f'question_type: {self.question_type}\n'
            f'ground_truth: {self.ground_truth}\n'
            f'generated_text: {self.generated_text}\n'
            f'parsed_answer: {self.parsed_answer}\n'
            f'confidence: {self.confidence}\n'
            f'hidden_states_len: {len(self.hidden_states)}'
        )
@dataclass(frozen=True)
class HiddenStateItem:
    """One layer from one saved inference result, used for memory-bounded loading."""
    image_id: int
    category: str
    question_type: str
    ground_truth: bool
    generated_text: str
    parsed_answer: bool | None
    confidence: float 
    layer_index: int
    hidden_state: np.ndarray

"""
Helper Methods
"""
def _model_device(model) -> torch.device:
    """Return the device on which the model parameters live."""
    return next(model.parameters()).device

def _built_prompt(processor, question: str) -> str:
    """Build the multimodal chat prompt for one image/question pair."""
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
    cleaned = re.sub(r'^assistant\s*:\s*', '', cleaned)

    match = re.fullmatch(
        r"""[\s"'`]*(yes|no)[\s"'`.!?,;:]*""",
        cleaned,
    )

    if match is None:
        return None

    return match.group(1) == 'yes'

def _fallback_parse_answer(text: str) -> bool | None:
    cleaned = text.strip().lower()

    has_yes = re.search(r'\byes\b', cleaned) is not None
    has_no = re.search(r'\bno\b', cleaned) is not None

    if has_yes and not has_no:
        return True
    if has_no and not has_yes:
        return False
    return None

def _parse_with_fallback(text: str) -> bool | None:
    strict_answer = _parse_answer(text)

    if strict_answer is not None:
        return strict_answer
    return _fallback_parse_answer(text)

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
    }

    if 'pixel_attention_mask' in inputs:
        kwargs['pixel_attention_mask'] = inputs['pixel_attention_mask']

    with torch.inference_mode():
        image_outputs = model.get_image_features(**kwargs)

    # Some Transformers version return the projected features directly.
    if torch.is_tensor(image_outputs):
        return image_outputs.detach()

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

def _answer_confidence(model, generation_output, generated_tokens_ids, processor) -> float:
    """
    Calculate the joint probability assigned to the generated answer token(s).
    """
    if generation_output.scores is None:
        raise RuntimeError('generate() did not return scores despite output_scores=true')

    transition_scores = model.compute_transition_scores(
        generation_output.sequences,
        generation_output.scores,
        normalize_logits=True
    )

    # Batch size is 1
    token_log_probs = transition_scores[0]

    tokenizer = getattr(processor, 'tokenizer', None)

    special_ids = set(tokenizer.all_special_ids) if tokenizer is not None else set()

    log_probabilities: list[torch.Tensor] = []

    for token_id_tensor, log_prob in zip(generated_tokens_ids, token_log_probs):
        token_id = int(token_id_tensor.item())

        if token_id in special_ids:
            continue

        log_probabilities.append(log_prob)

    if not log_probabilities:
        return 0.0

    total_log_prob = torch.stack(log_probabilities).sum()
    return float(torch.exp(total_log_prob).item())

def _to_storage_dtype(state: torch.Tensor) -> torch.Tensor:
    """Store floating hidden states as FP16 to keep raw cache manageable."""
    if state.is_floating_point() and state.dtype != torch.float16:
        state = state.to(torch.float16)
    return state

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
            if len(step_hidden_states) < num_layers:
                raise RuntimeError(
                    "Generation returned fewer hidden-state entries than the model has "
                    f"transformer layers: got {len(step_hidden_states)}, expected at least {num_layers}"
                )
            
            # Remove embedding state by taking the final num_layers entries
            transformer_states = step_hidden_states[-num_layers:]
            state = transformer_states[layer_index]

            if state.ndim != 3 or state.shape[0] != 1:
                raise RuntimeError(
                    'Expected generation hidden state with shape '
                    f'(1, seq_len, hidden_dim), got {tuple(state.shape)}'
                )

            # Remove the batch dimension since the batch size is 1.
            state = state[0].detach()
            state = _to_storage_dtype(state).cpu().contiguous()
            layer_parts.append(state)

        if not layer_parts:
            raise RuntimeError('generated() returned an empty hidden-state sequence')
        full_layer_sequence = torch.cat(
            layer_parts,
            dim=0
        )
        result[layer_index] = full_layer_sequence.numpy().copy()

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

    full_sequence = generation_output.sequences[0]

    generated_token_ids = full_sequence[prompt_length:]

    generated_text = processor.decode(
        generated_token_ids,
        skip_special_tokens=True,
    ).strip()

    parsed_answer = _parse_with_fallback(generated_text)

    confidence = _answer_confidence(
        model,
        generation_output,
        generated_token_ids,
        processor,
    )

    hidden_states = _convert_generation_hidden_states(model, generation_output.hidden_states)

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

def _validate_manifest_row(row: dict, index: int) -> None:
    """Fail early with a useful error when a required manifest field is missing."""
    required = {
        'image_id',
        'category',
        'question',
        'question_type',
        'ground_truth',
    }
    missing = sorted(required.difference(row))
    if missing:
        raise KeyError(f'Manifest row {index} is missing required fields: {missing}')

def _group_manifest(manifest: list[dict]) -> dict[int, list[tuple[int, dict]]]:
    grouped_rows: dict[int, list[tuple[int, dict]]] = {}

    for index, row in enumerate(manifest):
        _validate_manifest_row(row, index)
        image_id = int(row['image_id'])
        grouped_rows.setdefault(image_id, []).append((index, row))

    return grouped_rows

def _parse_result_metadata(result: InferenceResult) -> dict:
    strict_answer = _parse_answer(result.generated_text)

    return {
        'parsed_answer': result.parsed_answer,
        'used_fallback': strict_answer is None
    }

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
        'parse_result': _parse_result_metadata(result),
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
        'format': RESULT_FORMAT,
        'result_count': str(len(results))
    }

    for result_index, result in enumerate(results):
        layers = sorted(result.hidden_states.keys())

        for layer_index in layers:
            array = np.asarray(result.hidden_states[layer_index])

            if array.ndim != 2:
                raise ValueError(
                    f"Hidden state for result {result_index}, layer {layer_index} must be 2-D "
                    f"(seq_len, hidden_dim); got shape {array.shape}."
                )

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
        save_file(tensors, str(temp_path), metadata=metadata)

        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()

def _load_results_and_metadata(path: Path) -> tuple[list[InferenceResult], dict[str, str]]:
    """Load a SafeTensors results file created by _atomic_save_results"""
    results: list[InferenceResult] = []

    with safe_open(str(path), framework='pt', device='cpu') as file:
        metadata = file.metadata() or {}

        if metadata.get('format') != RESULT_FORMAT:
            raise RuntimeError(
                f'Unsupported or missing inference-results format in {path}: '
                f'{metadata.get('format')!r}'
            )

        result_count = int(metadata.get('result_count', '0'))

        for result_index in range(result_count):
            metadata_key = f'result_{result_index}'

            if metadata_key not in metadata_key:
                raise RuntimeError(f'Missing metadata entry {metadata_key!r} in {path}')

            item = json.loads(metadata[metadata_key])
            hidden_states: dict[int, np.ndarray] = {}

            for layer_index in item.get('layers', []):
                layer_index = int(layer_index)
                tensor_key = f'result_{result_index}.layer_{layer_index}'

                if tensor_key not in file.keys():
                    raise RuntimeError(f'Missing tensor {tensor_key!r} in {path}')

                tensor = file.get_tensor(tensor_key)
                tensor = _to_storage_dtype(tensor).cpu()
                hidden_states[layer_index] = tensor.numpy().copy()

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

def _array_nbytes_from_shape(shape: list[int] | tuple[int, ...], dtype: str) -> int:
    """Calculate tensor size without loading the tensor."""
    dtype_sizes = {
        'BOOL': 1,
        'U8': 1,
        'I8': 1,
        'I16': 2,
        'F16': 2,
        'BF16': 2,
        'I32': 4,
        'F32': 4,
        'I64': 8,
        'F64': 8,
    }

    if dtype not in dtype_sizes:
        raise ValueError(f'Unsupported dtype: {dtype}')

    num_values = 1
    for dimenstion in shape:
        num_values *= dimenstion

    return num_values * dtype_sizes[dtype]

def _iter_hidden_state_batches(checkpoint_dir: str | Path, max_batch_bytes: int = MAX_BATCH_BYTES):
    """
    Load multiple hidden-state tensors into RAM at once, 
    while keeping the total hidden-state payload of each batch under max_batch_bytes.

    Yields:
        list[HiddenStateItem]

    Each batch may contain several examples/layers.

    """
    if max_batch_bytes <= 0:
        raise ValueError('max_batch_bytes must be positive')
    
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_paths = sorted(checkpoint_dir.glob('row_*.safetensors'))

    batch: list[HiddenStateItem] = []
    batch_bytes = 0

    for checkpoint_path in checkpoint_paths:
        with safe_open(str(checkpoint_path), framework='pt', device='cpu') as file:
            metadata = file.metadata() or {}

            if metadata.get('format') != RESULT_FORMAT:
                raise RuntimeError(f'Unsupported checkpoint format in {checkpoint_path}')
            
            if int(metadata.get('result_count', '0')) != 1:
                raise RuntimeError(f'Expected one result in {checkpoint_path}')

            if 'result_0' not in metadata:
                raise RuntimeError('Missing result_0 metadata in {checkpoint_path}')

            item = json.loads(metadata['result_0'])

            for layer_index in item['layers']:
                layer_index = int(layer_index)
                tensor_key = f"result_0.layer_{layer_index}"
                tensor_slice = file.get_slice(tensor_key)
                shape = tensor_slice.get_shape()
                dtype = tensor_slice.get_dtype()
                tensor_bytes = _array_nbytes_from_shape(shape, dtype)

                # Return to current batch first if adding current batch exceeds budget
                if (batch and batch_bytes + tensor_bytes > max_batch_bytes):
                    yield batch
                    batch = []
                    batch_bytes = 0

                # Materialise the whole hidden-state tensor
                tensor = file.get_tensor(tensor_key)
                tensor = _to_storage_dtype(tensor).cpu()
                hidden_state = tensor.numpy().copy()

                result_item = HiddenStateItem(
                    image_id=int(item['image_id']),
                    category=str(item['category']),
                    question_type=str(item['question_type']),
                    ground_truth=bool(item['ground_truth']),
                    generated_text=str(item['generated_text']),
                    parsed_answer=item['parsed_answer'],
                    confidence=float(item['confidence']),
                    layer_index=int(layer_index),
                    hidden_state=hidden_state,
                )

                batch.append(result_item)
                batch_bytes += tensor_bytes

    if batch:
        yield batch

def _checkpoint_path(checkpoint_dir: Path, manifest_index: int, image_id: int) -> Path:
    """One checkpoint file per QA pair."""
    return checkpoint_dir / (
            f"row_{manifest_index:06d}_image_{image_id:012d}.safetensors"
    )


def _checkpoint_is_valid(path: Path, expected_manifest_index: int | None = None, expected_image_id: int | None = None) -> bool:
    """Load a valid checkpoint"""
    if not path.exists(): 
        return None

    try:
        with safe_open(str(path), framework='pt', device='cpu') as file:
            metadata = file.metadata() or {}

            if metadata.get('format') != 'vlm_inference_results_v1':
                return False

            if int(metadata.get('result_count', '0')) != 1:
                return False

            if expected_manifest_index is not None:
                if metadata.get('manifest_index') != str(expected_manifest_index):
                    return False

            if 'result_0' not in metadata:
                return False

            item = json.loads(metadata['result_0'])

            if expected_image_id is not None and int(item.get('image_id', -1)) != expected_image_id:
                return False
            
            layers = item.get('layers')
            if not isinstance(layers, list):
                return False

            tensors_keys = set(file.keys())
            # Confirm every expected hidden-state tensor exists.
            for layer_index in layers:
                expected_key = f"result_0.layer_{layer_index}"
                if expected_key not in tensors_keys:
                    return False

                shape = file.get_slice(expected_key).get_shape()
                if len(shape) != 2 or any(int(dimension) <= 0 for dimension in shape):
                    return False

        return True
    except Exception:
        # A bad checkpoint should not destroy the run.
        return False

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

def _run_manifest_in_memory(model, processor, manifest: list[dict], image_dir: str) -> list[InferenceResult]:
    """
    Memory-safe full-run inference.
    Each completed QA pair is written immediately to its SafeTensors checkpoint and released from RAM.
    """
    if not manifest: 
        return []

    # Group rows by image to run the vision encoder once
    grouped_rows = _group_manifest(manifest)
    ordered_results: list[InferenceResult | None] = [None] * len(manifest)

    with tqdm(total=len(manifest), desc="Running inference", unit='QA') as progress:
        for image_id, image_rows in grouped_rows.items():
            first_row = image_rows[0][1]
            path = _image_path(image_dir, image_id, first_row)

            with Image.open(path) as opened_image:
                image = opened_image.convert('RGB')

            # cache vision output ONCE for this image
            image_hidden_states = None
            if hasattr(model, 'get_image_features'):
                first_inputs = _prepare_inputs(
                    model,
                    processor,
                    image,
                    str(first_row['question']),
                )
                image_hidden_states = _extract_image_hidden_states(model, first_inputs)
                del first_inputs

            # Run unfinished questions.
            for original_index, row in image_rows:
                result = _run_single_example_implementation(
                    model,
                    processor,
                    image,
                    str(row['question']),
                    image_hidden_states=image_hidden_states,
                )

                # Add metadata that run_single_example cannot know
                ordered_results[original_index] = replace(
                    result,
                    image_id=image_id,
                    category=str(row['category']),
                    question_type=str(row['question_type']),
                    ground_truth=_coerce_ground_truth(row['ground_truth']),
                )

                progress.update(1)

            # The cached feature tensor is no longer needed after the three questions for this image
            del image_hidden_states
            del image

    if any(result is None for result in ordered_results):
        raise RuntimeError('Inference failed to produce one result per manifest row.')

    return [result for result in ordered_results if result is not None]

def _run_manifest_resumable(model, processor, manifest: list[dict], image_dir: str, checkpoint_dir: Path) -> list[Path]:
    """
    Memory-safe full-run inference.

    Each completed QA pair is atomically checkpointed and immediately released from RAM.
    Existing structurally valid checkpoints are skipped on resumed runs.
    """
    if not manifest:
        return []

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    grouped_rows = _group_manifest(manifest)
    checkpoint_paths: list[Path | None] = [None] * len(manifest)

    with tqdm(total=len(manifest), desc='Full inference', unit='QA') as progress:
        for image_id, image_rows in grouped_rows.items():
            missing_rows: list[tuple[int, dict]] = []

            for original_index, row in image_rows:
                checkpoint = _checkpoint_path(checkpoint_dir, original_index, image_id)

                if _checkpoint_is_valid(checkpoint, expected_manifest_index=original_index, expected_image_id=image_id):
                    checkpoint_paths[original_index] = checkpoint

                    # Exisiting checkpoint counts as already completed.
                    progress.update(1)
                else:
                    missing_rows.append((original_index, row))

            if not missing_rows:
                continue

            first_missing_row = missing_rows[0][1]
            path = _image_path(image_dir, image_id, first_missing_row)

            with Image.open(path) as opened_image:
                image = opened_image.convert('RGB')

            image_hidden_states = None

            if hasattr(model, 'get_image_features'):
                first_inputs = _prepare_inputs(model, processor, image, str(first_missing_row['question']))
                image_hidden_states = _extract_image_hidden_states(model, first_inputs)
                del first_inputs

            for original_index, row in missing_rows:
                result = _run_single_example_implementation(
                    model,
                    processor,
                    image,
                    str(row['question']),
                    image_hidden_states=image_hidden_states,
                )

                result = replace(
                    result,
                    image_id=image_id,
                    category=str(row['category']),
                    question_type=str(row['question_type']),
                    ground_truth=_coerce_ground_truth(row['ground_truth']),
                )

                checkpoint = _checkpoint_path(checkpoint_dir, original_index, image_id)
                _save_checkpoint(result, checkpoint, original_index)
                checkpoint_paths[original_index] = checkpoint

                # Avoid retaining a 15+ GB result list in RAM
                del result

                progress.update(1)

            del image_hidden_states
            del image

    if any(path is None for path in checkpoint_paths):
        raise RuntimeError('Inference failed to produce one checkpoint per manifest row.')

    return [path for path in checkpoint_paths if path is not None]

def load_checkpoint_metadata(checkpoint_paths: list[str | Path]) -> list[dict]:
    results = []

    for path in checkpoint_paths:
        with safe_open(str(path), framework='pt', device='cpu') as file:
            metadata = file.metadata() or {}

            if 'result_0' not in metadata:
                raise RuntimeError(f'Missing result_0 metadata in {path}')

            results.append(json.loads(metadata['result_0']))

    return results

def load_model(model_name: str, device: str) -> tuple[PreTrainedModel, ProcessorMixin]:
    if device.startswith('cuda') and not torch.cuda.is_available():
        device = 'cpu'      # Fallback

    processor: ProcessorMixin = AutoProcessor.from_pretrained(model_name)

    if device.startswith('cuda'):
        model_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    elif device.startswith('mps'):
        model_dtype = torch.float16
    else:
        model_dtype = torch.float32

    model: PreTrainedModel = _AutoModel.from_pretrained(
        model_name, 
        dtype=model_dtype,
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
    return _run_manifest_in_memory(model, processor, manifest, image_dir)

def run_inference_resumable(model, processor, manifest: list[dict], image_dir: str, checkpoint_dir: str) -> list[Path]:
    """
    Run inference with one atomic SafeTensors checkpoint per QA pair.
    Exisiting valid checkpoints are skipped. The returneed paths preserve manifest order.
    """
    return _run_manifest_resumable(model, processor, manifest, image_dir, checkpoint_dir=Path(checkpoint_dir))

def iter_hidden_state_batches(checkpoint_dir: str | Path, max_batch_bytes: int = MAX_BATCH_BYTES):
    """Public memory-bounded iterator over resumable checkpoint hidden states."""
    yield from _iter_hidden_state_batches(checkpoint_dir, max_batch_bytes)

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

