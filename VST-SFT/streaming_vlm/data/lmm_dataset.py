from dataclasses import dataclass, field
import json, torch, random, tqdm, io, functools,os
from PIL import Image
from torch.utils.data import Dataset
from transformers import logging, AutoProcessor, AutoModel, Qwen2_5_VLForConditionalGeneration
from torchvision.transforms.functional import pil_to_tensor
from transformers.feature_extraction_utils import BatchFeature
from collections import defaultdict
from livecc_utils import _read_video_decord_plus, _spatial_resize_video
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import List
from qwen_vl_utils.vision_process import smart_nframes, process_vision_info, FPS, VIDEO_TOTAL_PIXELS, VIDEO_MIN_PIXELS, FPS_MAX_FRAMES, FORCE_QWENVL_VIDEO_READER
import sys 
sys.path.append('/workspace/images-ks3-hd/workspace/yinliang/code/streaming-vlm')
from streaming_vlm.utils.get_qwen_range import get_qwen_range
from transformers import set_seed

class mute_stderr_ffmpeg:
    def __enter__(self):
        self._stderr_fd = os.dup(2)
        self._devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(self._devnull, 2)
        return self
    def __exit__(self, *exc):
        os.dup2(self._stderr_fd, 2)
        os.close(self._devnull)
        os.close(self._stderr_fd)

logger = logging.get_logger(__name__)

@dataclass
class DataArguments:
    train_annotation_paths: list[str] = None
    train_write_traj_paths: list[str] = None
    initial_fps_frames: int = int(FPS)
    streaming_fps_frames: int = int(FPS)
    with_context: bool = False
    text_sink:int = 0
    text_sliding_window:int = 0

@dataclass
class EvalDataArguments:
    eval_annotation_paths: list[str] = None
    eval_write_traj_paths: list[str] = None

def readlastline(path: str):
    """Efficiently read the last line of a file."""
    with open(path, "rb") as f:
        f.seek(-2, 2)
        while f.read(1) != b"\n":
            f.seek(-2, 1)
        return f.readline()

def bytes_to_pil(image_bytes):
    """Convert bytes to a PIL Image object."""
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode == 'P':
        image = image.convert('RGBA')
    return image.convert('RGB')

def get_phrase_before_timestamp(text_stream, timestamp, start_from: int = 0):
    """From a word stream with timestamps, extract the phrase consisting of all words whose end time is before the given timestamp."""
    phrase = ''
    i = 0
    for i, (ws, we, word) in enumerate(text_stream[start_from:]):
        if timestamp >= we:
            phrase += ' ' + word.strip()
            if i == len(text_stream[start_from:]) - 1:
                i += 1
                break
        else:
            break
    return phrase, i + start_from



class LMMDataset(Dataset):
    """PyTorch dataset for multimodal large language models."""
    def __init__(
        self, *, train_annotation_paths: list[str] = None, eval_annotation_paths: list[str] = None, processor: AutoProcessor, tokenizer = None,
        initial_fps_frames: int = DataArguments.initial_fps_frames, streaming_fps_frames: int = DataArguments.streaming_fps_frames, 
        with_context: str = DataArguments.with_context, return_conversation: bool = False,text_sink:int = DataArguments.text_sink,
        text_sliding_window:int = DataArguments.text_sliding_window,
        **kwargs
    ):
        super().__init__()
        self.return_conversation = return_conversation
        self.handles = []
        if eval_annotation_paths is not None:
            self.is_eval_dataset = True
            annotation_paths = eval_annotation_paths
        else:
            self.is_eval_dataset = False
            annotation_paths = train_annotation_paths
            
        for annotation_path in annotation_paths:
            assert annotation_path.endswith('.jsonl'), "Please organize annotation data as JSONL (one sample per line) and store the final line as the seek index."
            root, fname = os.path.split(annotation_path)
            stem = fname.replace("_with_seeks", "").rsplit(".jsonl", 1)[0]
            seek_path = os.path.join(root, f"{stem}_seeks.jsonl")

            logger.warning(f"Loading {annotation_path}")
            logger.warning(f"Loading seek index from {seek_path}")

            with open(seek_path) as f:
                seeks = json.load(f)
                
            self.handles.extend(zip([annotation_path] * len(seeks), seeks))
            logger.warning(f"Successfully loaded {annotation_path}")

        if 'Qwen2VL' in processor.__class__.__name__:
            self.im_start_id, self.assistant_id, self.newline_id, self.im_end_id = processor.tokenizer('<|im_start|>assistant\n<|im_end|>').input_ids
            self.get_range = get_qwen_range
            self.model_base = 'Qwen2'
        elif 'Qwen2_5_VL' in processor.__class__.__name__:
            self.im_start_id, self.assistant_id, self.newline_id, self.im_end_id = processor.tokenizer('<|im_start|>assistant\n<|im_end|>').input_ids
            self.get_range = get_qwen_range
            self.model_base = 'Qwen2'
        else:
            raise NotImplementedError(f"Video preprocessing for {processor.__class__.__name__} is not implemented")

        self.processor = processor
        self.with_context = with_context
        self.initial_fps_frames = initial_fps_frames
        self.streaming_fps_frames = streaming_fps_frames
        self.text_sink = text_sink
        self.text_sliding_window = text_sliding_window
    
    def load_conversation(self, index):
        """Load a single conversation by index."""
        annotation_path, seek = self.handles[index]
        with open(annotation_path) as f:
            f.seek(seek)
            line = f.readline()
        line = json.loads(line)
        return line

    def preprocess_image(self, element: dict):
        """Preprocess image data."""
        if hasattr(self, 'remote_loader'):
            return Image.open(self.remote_loader(element['image']))
        return element['image']
    
    def preprocess_video(self, element: dict):
        """Preprocess video data."""
        if 'pos' in element:
            positions = [0] + element['pos']
            nframes = smart_nframes(element, total_frames=len(positions) - 1, video_fps=FPS)
            sampler = torch.linspace(0, len(positions) - 2, nframes).round().long()
            data_bytes = self.remote_loader(element['video'], length_check=True, return_io=False)
            video = torch.stack([pil_to_tensor(bytes_to_pil(data_bytes[positions[i]:positions[i+1]])) for i in sampler])
            video = _spatial_resize_video(video)
            return video
        return element['video']

    def preprocess_text(self, element: str):
        """Preprocess text data."""
        return element['text']

    def preprocess_conversation_stream(self, conversation: list):
        """Simulate converting timestamped conversation into a streaming multi-turn dialogue."""
        user_message, assistant_message = conversation
        user_content, assistant_content = user_message['content'], assistant_message['content']

        user_video_dict, user_query_dict = user_content
        video_start = user_video_dict['video_start']
        video_end = user_video_dict['video_end']
        
        assert 'video' in user_video_dict, 'Please check your data: the first user content must contain video information'

        assistant_text_stream = assistant_message['content'][0]['text_stream']
        qa_stream = assistant_message['content'][0]['qa_stream'] if 'qa_stream' in assistant_message['content'][0] else []
        
        with mute_stderr_ffmpeg():
            clip, _, clip_pts = _read_video_decord_plus(
                user_video_dict, return_pts=True, strict_fps=True
            )
        clip = _spatial_resize_video(clip)

        start_timestamp, end_timestamp = video_start, video_start + self.initial_fps_frames / FPS

        phrase, next_start_from = get_phrase_before_timestamp(
            assistant_text_stream,
            end_timestamp
        )

        if len(qa_stream) > 0 and start_timestamp < qa_stream[0][1] and end_timestamp >= qa_stream[0][1]:
            question = qa_stream[0][2]
            answer = qa_stream[0][3]
            qa_stream = qa_stream[1:]
        else:
            question = ''
            answer = ''
        
        user_content = [
                        {'type': 'text',  'text': f'Time={start_timestamp:.1f}-{end_timestamp:.1f}s' + f'{question}'},
                        {'type': 'video', 'video': clip[:self.initial_fps_frames]},
                    ]
        assistant_content = [{'type': 'text', 'text': answer + '\n' + phrase + ' ...'}]
        conversation = [
            {
                'role': 'user',
                'content': user_content
            },
            {
                'role': 'assistant',
                'content': assistant_content
            }
        ] 
        frames_list = [clip[:self.initial_fps_frames]]
        
        for i in range(self.initial_fps_frames, len(clip), self.streaming_fps_frames):
            start_timestamp, end_timestamp = video_start + i / FPS, video_start + (i + self.streaming_fps_frames) / FPS
            
            phrase, next_start_from = get_phrase_before_timestamp(
                assistant_text_stream,
                end_timestamp,
                start_from=next_start_from
            )
            if len(qa_stream) > 0 and start_timestamp < qa_stream[0][1] and end_timestamp >= qa_stream[0][1]:
                question = qa_stream[0][2]
                answer = qa_stream[0][3]
                qa_stream = qa_stream[1:]
            else:
                question = ''
                answer = ''
            
            frames = clip[i : i + self.streaming_fps_frames]

            user_content = [
                    {'type': 'text',  'text': f'Time={start_timestamp:.1f}-{end_timestamp:.1f}s {question}'},
                    {'type': 'video', 'video': frames},
                ]
            assistant_content = [{'type': 'text', 'text': answer + '\n' + phrase + ' ...'}]

            conversation.extend([
                {
                    'role': 'user',
                    'content': user_content
                },
                {
                    'role': 'assistant',
                    'content': assistant_content
                }
            ])
            frames_list.append(frames)

        return conversation, frames_list

    def getitem(self, index, return_text=False):
        """Core logic to get and preprocess a single data item."""
        conversation = self.load_conversation(index)


        special_process_for_stream, image_inputs, video_inputs = False, None, None
        previous_text = ''
        for message in conversation:
            if message['role'] == 'user':
                for element in message['content']:
                    if 'previous' in element:
                        previous_text = element['previous']
                        element['previous'] = ''
                    if hasattr(self, 'remote_loader'):
                        element['remote_loader'] = self.remote_loader
                    modal = element['type']
                    element[modal] = getattr(self, f'preprocess_{modal}')(element)
                    if isinstance(element[modal], torch.Tensor):
                        if video_inputs is None:
                            video_inputs = [element[modal]]
                        else:
                            video_inputs.append(element[modal])
            else:
                for element in message['content']:
                    special_process_for_stream = 'text_stream' in element
                    break

        
        if not os.path.exists(conversation[0]['content'][0]['video']):
            if os.path.exists(os.path.join(os.environ['DATASET_PATH'], conversation[0]['content'][0]['video'])):    
                conversation[0]['content'][0]['video'] = os.path.join(os.environ['DATASET_PATH'], conversation[0]['content'][0]['video'])
            else:
                raise ValueError(f"Video {conversation[0]['content'][0]['video']} not found")
        
        if special_process_for_stream:
            conversation, video_inputs = self.preprocess_conversation_stream(conversation)
            image_inputs = None
        else:
            if not video_inputs and not image_inputs:
                image_inputs, video_inputs = process_vision_info(conversation)

        conversation = [{"role": "previous text", "content": previous_text}] + conversation

        if return_text:
            return conversation
        texts = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False, return_tensors='pt')
        
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
        )

        if self.text_sink != 0 or self.text_sliding_window != 0:
            previous_text_start_idx, previous_text_end_idx = self.get_range(inputs.input_ids, 'previous text', 0, contain_lf=True)
            need_truncate = previous_text_start_idx + self.text_sink + self.text_sliding_window <= previous_text_end_idx + 1
            if need_truncate:
                truncate_start = previous_text_start_idx + self.text_sink
                truncate_end = previous_text_end_idx - self.text_sliding_window
                inputs['input_ids'] = torch.cat([inputs.input_ids[:,:truncate_start],inputs.input_ids[:,truncate_end+1:]],dim=1).contiguous()
                if 'attention_mask' in inputs:
                    inputs['attention_mask'] = torch.cat([inputs.attention_mask[:,:truncate_start],inputs.attention_mask[:,truncate_end+1:]],dim=1).contiguous()

        input_ids = inputs.input_ids
        labels = torch.full_like(input_ids, fill_value=-100, dtype=input_ids.dtype)
        im_start_idxs = (input_ids == self.im_start_id).nonzero()
        im_end_idxs = (input_ids == self.im_end_id).nonzero()
        for (sample_idx, im_start_idx), (sample_idx, im_end_idx) in zip(im_start_idxs, im_end_idxs):
            if input_ids[sample_idx, im_start_idx + 1] == self.assistant_id:
                labels[sample_idx, im_start_idx+3:im_end_idx+1] = input_ids[sample_idx, im_start_idx+3:im_end_idx+1]

        inputs['labels'] = labels
        if self.return_conversation:
            inputs['conversation'] = conversation
            inputs['start_timestamp'] = conversation[0]['content'][0]['video_start']

        return inputs

    # def __getitem__(self, index):
    #     """Dataset standard method with retry."""
    #     return self.getitem(index) 

    #     try: 
    #         return self.getitem(index) 
    #     except Exception as e:
    #         logger.warning(
    #             f"{'Training' if not self.is_eval_dataset else 'Eval'}: bug at video: "
    #             f"{self.load_conversation(index)[0]['content'][0]['video']}"
    #             f"{e}"
    #         )
    #     return self.__getitem__( index*13 %len(self.handles))

    def __getitem__(self, index):

        """Dataset standard method with retry and error logging."""

        try:

            # 确保这里只调用一次 getitem

            return self.getitem(index)

        except Exception as e:

            # 获取当前出问题的视频路径

            try:

                raw_data = self.load_conversation(index)

                video_path = raw_data[0]['content'][0].get('video', 'Unknown Path')

            except:

                video_path = f"Could not load path for index {index}"



            error_msg = f"ERROR at index {index}, video: {video_path}. Exception: {str(e)}"

            

            # 1. 直接打印到终端

            print("\n" + "="*50)

            print(error_msg)

            print("="*50 + "\n")

            





            # 3. 递归重试：随机换一个索引继续，保证程序不中断

            return self.__getitem__((index + 1) % len(self.handles))

    def data_collator(self, batched_inputs, **kwargs):
        assert len(batched_inputs) == 1, "batch size must be 1"
        return batched_inputs[0]

    def __len__(self):
        """Return the total number of samples."""
        return len(self.handles)


import os
import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
from typing import Optional
from qwen_vl_utils.vision_process import (
    FORCE_QWENVL_VIDEO_READER, VIDEO_TOTAL_PIXELS, FPS_MAX_FRAMES, VIDEO_MIN_PIXELS, VIDEO_MAX_PIXELS, FRAME_FACTOR, IMAGE_FACTOR, FPS,
    smart_nframes, smart_resize
)

# 假设这些是你原本代码中的全局常量或函数，这里为了代码不报错进行占位说明
# 请确保你的环境中可以访问这些变量/函数

def _read_video_images_list(ele: dict, strict_fps: bool = False, drop_last: bool = True, return_pts: bool = False, only_get_last_frame: Optional[int] = None):
    """
    Read video from a list of image paths (Frame Sequence).
    Compatible with _read_video_decord_plus inputs and outputs.
    """
    
    # 1) Parse Paths
    image_paths = ele["video"]
    dataset_path = os.environ.get('DATASET_PATH', '')
    
    # 检查第一张图路径，决定是否拼接 DATASET_PATH
    if not os.path.exists(image_paths[0]):
        # 尝试拼接
        full_image_paths = [os.path.join(dataset_path, p) for p in image_paths]
    else:
        full_image_paths = image_paths

    total_frames = len(full_image_paths)
    if total_frames == 0:
        raise ValueError("Input image list is empty.")

    # 2) Determine FPS and PTS (Time info)
    # 图片列表没有元数据，必须依赖 video_start/end 或默认假设
    video_start = ele.get('video_start', 0.0)
    video_end = ele.get('video_end', None)
    
    if video_end is None:
        # 如果没有结束时间，假设一个默认 FPS (例如 30) 来计算 pts
        assumed_fps = 30.0
        video_end = video_start + (total_frames / assumed_fps)
        video_fps = assumed_fps
    else:
        # 如果有起止时间，根据帧数倒推 FPS
        duration = video_end - video_start
        if duration <= 0:
            duration = 1e-6 # 避免除零
        video_fps = total_frames / duration

    # 生成每一帧对应的时间戳
    # 假设图片是均匀分布在 [start, end] 区间
    video_pts = np.linspace(video_start, video_end, total_frames, endpoint=False)
    
    # 3) Sampling strategy (Logic copied from _read_video_decord_plus)
    clip_idxs = None
    
    # 注意：对于图片列表，通常列表本身就是 crop 过的片段，
    # 所以不需要像 decord 那样先根据 start/end 筛选一遍 indices，
    # 这里的 video_pts 已经对应了 full_image_paths 的每一项。

    if not strict_fps:
        # Adaptive sampling
        nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
        # 均匀采样索引
        clip_idxs = np.linspace(0, total_frames - 1, nframes).round().astype(int)
    else:
        # Fixed FPS sampling
        # 生成基于全局 FPS 的期望时间戳
        expected_timestamps = np.arange(video_pts[0], video_pts[-1] + 1e-6, 1 / FPS)

        if len(expected_timestamps) > FPS_MAX_FRAMES:
            if drop_last:
                expected_timestamps = expected_timestamps[:FPS_MAX_FRAMES]
            else:
                expected_timestamps = expected_timestamps[
                    np.linspace(0, len(expected_timestamps) - 1, FPS_MAX_FRAMES).round().astype(int)
                ]

        # 找到最接近期望时间戳的帧索引
        # expected_timestamps[:, None] shape: (M, 1)
        # video_pts shape: (N,)
        # 广播比较，找到每一行(期望时间)在 video_pts 中最接近的索引
        # 注意：这里假设 video_pts 是单调递增的
        expected_idxs_for_clip_pts = np.abs(expected_timestamps[:, None] - video_pts).argmin(axis=1)
        
        clip_idxs = expected_idxs_for_clip_pts.tolist()
        
        # 对应的实际时间戳
        clip_pts_sampled = video_pts[clip_idxs].tolist()

        # Padding logic
        while len(clip_idxs) % FRAME_FACTOR != 0:
            clip_idxs.append(clip_idxs[-1])
            clip_pts_sampled.append(clip_pts_sampled[-1])

    # Handle only_get_last_frame
    if only_get_last_frame:
        clip_idxs = clip_idxs[-only_get_last_frame:]
        if strict_fps:
             clip_pts_sampled = clip_pts_sampled[-only_get_last_frame:]

    # 4) Load Frames
    # 这一步替代 vr.get_batch
    loaded_frames = []
    for idx in clip_idxs:
        img_path = full_image_paths[idx]
        try:
            # 使用 PIL 读取并转为 Tensor (C, H, W) 范围 [0.0, 1.0]
            # 如果原函数 decord 返回的是 [0, 255] 的 uint8，这里可能需要调整
            # 通常 decord.asnumpy() 返回 uint8 [0-255]，而 to_tensor 返回 float [0-1]
            # 为了保持一致性，这里我们模拟 decord 的 uint8 行为
            
            img = Image.open(img_path).convert('RGB')
            img_np = np.array(img) # (H, W, C) uint8
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1) # (C, H, W)
            loaded_frames.append(img_tensor)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # 简单的错误处理：复制上一帧或生成黑帧
            if len(loaded_frames) > 0:
                loaded_frames.append(loaded_frames[-1])
            else:
                # 假设 224x224，实际应根据需求调整
                loaded_frames.append(torch.zeros(3, 224, 224, dtype=torch.uint8))

    # Stack into (T, C, H, W)
    clip = torch.stack(loaded_frames) 

    # Calculate effective sample FPS
    sample_fps = len(clip_idxs) / max(total_frames, 1e-6) * video_fps

    # 5) Return
    if return_pts:
        # 如果是 strict_fps，我们在上面已经算好了 clip_pts_sampled
        # 如果不是，我们需要根据 clip_idxs 从 video_pts 取
        if not strict_fps:
            clip_pts_sampled = video_pts[clip_idxs].tolist()
            
        return clip, sample_fps, clip_pts_sampled
        
    return clip, sample_fps


class streamingDataset(LMMDataset):
    # def frame_sample(self, clip, start_timestamp, end_timestamp):

    def preprocess_conversation_stream(self, conversation: list):
        """
            Simulate converting timestamped conversation into a streaming multi-turn dialogue.
            We support two formats: LiveCC and our StreamingThinking.
            - For LiveCC, we use a fixed number of frames. 
            - For ours, we use a Pyscene-detect-based CLIP method to simulate the streaming process.
            The program first reads the conversation, if duration of two consecutive turns is larger than self.streaming_fps_frames, we use the time-clock directly. Otherwise, we use t0+self.streaming_fps_frames.
        """
        # SFT说明：
        # 这里是当前 VST 风格 SFT 的核心数据展开步骤。
        # 一条原始样本会被改写成按时间切开的多轮 user/assistant 对话，
        # 每一轮都对齐一个局部视频 clip。
        # 监督目标仍然是 assistant 文本，而不是 memory action。
        user_message, assistant_message = conversation
        user_content, assistant_content = user_message['content'], assistant_message['content']


        user_video_dict, user_query_dict = user_content
        video_start = user_video_dict['video_start']
        video_end = user_video_dict['video_end']
        
        assert 'video' in user_video_dict, 'Please check your data: the first user content must contain video information'

        assistant_text_stream = assistant_message['content'][0]['text_stream']
        qa_stream = assistant_message['content'][0]['qa_stream'] if 'qa_stream' in assistant_message['content'][0] else []
        clip_num = len(assistant_text_stream) + len(qa_stream)
        with mute_stderr_ffmpeg():
            if type(user_video_dict['video']) == str:
                clip, _, clip_pts = _read_video_decord_plus(
                    user_video_dict, return_pts=True, strict_fps=True
                )
            else:
                clip, _, clip_pts = _read_video_images_list(
                    user_video_dict, return_pts=True, strict_fps=True
                )
        clip = _spatial_resize_video(clip)

        start_timestamp, end_timestamp = video_start, video_start + self.initial_fps_frames / FPS
        
        phrase, next_start_from = get_phrase_before_timestamp(
            assistant_text_stream,
            end_timestamp
        )

        conversation = [] 
        frames_list = []

        i = 0
        clip_start_idx = 0
        clip_end_idx = 0
        # ours，采样比较稀疏，时间戳直接用assistant_text_stream/qa_stream
        instruction = user_query_dict['text']
        while i < len(assistant_text_stream):
            start_timestamp, end_timestamp, phrase = assistant_text_stream[i]
            clip_start_idx = int(len(clip)*(start_timestamp-video_start)/(video_end-video_start))
            clip_end_idx = int(len(clip)*(end_timestamp-video_start)/(video_end-video_start))
            
            answer = ''
            
            i += 1
            if clip_end_idx > clip_start_idx:
                frames = clip[clip_start_idx : clip_end_idx]
            else:
                frames = None

            if phrase == '':
                phrase = ''
            
            user_content = [ {'type': 'text',  'text': f'Time={start_timestamp:.1f}-{end_timestamp:.1f}s '},]
            if frames != None:
                user_content.append({'type': 'video', 'video': frames})

            user_content.append(
                    {'type': 'text',  'text': f'{instruction}'},
            )
            
            assistant_content = [{'type': 'text', 'text': answer + phrase }]

            # SFT说明：
            # 每个 clip 会被变成一轮训练样本：
            # user = 当前时间戳 + 当前视频片段 + instruction
            # assistant = 这一段对应的文本监督
            # 这就是 VST 风格的 sequence-style intermediate supervision。

            conversation.extend([
                {
                    'role': 'user',
                    'content': user_content
                },
                {
                    'role': 'assistant',
                    'content': assistant_content
                }
            ])
            if frames != None:
                frames_list.append(frames)

        i = 0
        while i < len(qa_stream):
            start_timestamp, end_timestamp, _, __ = qa_stream[i]
            clip_start_idx = int(len(clip)*(start_timestamp-video_start)/(video_end-video_start))
            clip_end_idx = int(len(clip)*(end_timestamp-video_start)/(video_end-video_start))

            question = qa_stream[i][2]
            answer = qa_stream[i][3]    
            phrase = ''
            i += 1

            if clip_end_idx > clip_start_idx:
                frames = clip[clip_start_idx : clip_end_idx]
            else:
                frames = None

            if phrase == '':
                phrase = ''
            
            if frames != None:
                user_content = [{'type': 'video', 'video': frames}]
            else:
                user_content = []
            
            user_content.append(
                    {'type': 'text',  'text': f'Timestamp={start_timestamp:.1f}s {question}'},
            )
            
            assistant_content = [{'type': 'text', 'text': answer + phrase }]

            # SFT说明：
            # qa_stream 会在 clip-text stream 之外，再补充显式 QA 轮次。
            # 但训练目标仍然是 assistant 文本生成，不是动作分类。
            conversation.extend([
                {
                    'role': 'user',
                    'content': user_content
                },
                {
                    'role': 'assistant',
                    'content': assistant_content
                }
            ])
            if frames != None:
                frames_list.append(frames)       

        return conversation, frames_list
    
    def getitem(self, index, return_text=False):
        """Core logic to get and preprocess a single data item."""
        conversation = self.load_conversation(index)

        special_process_for_stream, image_inputs, video_inputs = False, None, None
        previous_text = ''
        for message in conversation:
            if message['role'] == 'user':
                for element in message['content']:
                    if 'previous' in element:
                        question = element['text']
                        previous_text = element['previous']
                        element['previous'] = ''
                    if hasattr(self, 'remote_loader'):
                        element['remote_loader'] = self.remote_loader
                    modal = element['type']
                    element[modal] = getattr(self, f'preprocess_{modal}')(element)
                    if isinstance(element[modal], torch.Tensor):
                        if video_inputs is None:
                            video_inputs = [element[modal]]
                        else:
                            video_inputs.append(element[modal])
            else:
                for element in message['content']:
                    special_process_for_stream = 'text_stream' in element
                    break

        
        # if not os.path.exists(conversation[0]['content'][0]['video']):
        #     if os.path.exists(os.path.join(os.environ['DATASET_PATH'], conversation[0]['content'][0]['video'])):    
        #         conversation[0]['content'][0]['video'] = os.path.join(os.environ['DATASET_PATH'], conversation[0]['content'][0]['video'])
        #     else:
        #         raise ValueError(f"Video {conversation[0]['content'][0]['video']} not found")

        # 获取 video 字段
        video_content = conversation[0]['content'][0]['video']
        dataset_root = os.environ.get('DATASET_PATH', '') # 安全获取环境变量

        # 情况 1: video 是一个列表 (多帧图片)
        if isinstance(video_content, list):
            # 遍历列表中的每一帧路径进行处理
            new_video_list = []
            for frame_path in video_content:
                # 检查绝对路径是否存在
                if os.path.exists(frame_path):
                    new_video_list.append(frame_path)
                # 检查拼接 DATASET_PATH 后是否存在
                elif os.path.exists(os.path.join(dataset_root, frame_path)):
                    new_video_list.append(os.path.join(dataset_root, frame_path))
                else:
                    # 如果找不到，抛出异常，提示具体是哪一帧
                    raise ValueError(f"Video frame not found: {frame_path}")
            
            # 更新回 conversation
            conversation[0]['content'][0]['video'] = new_video_list

        # 情况 2: video 是一个字符串 (单个视频文件)
        elif isinstance(video_content, str):
            if not os.path.exists(video_content):
                if os.path.exists(os.path.join(dataset_root, video_content)):
                    conversation[0]['content'][0]['video'] = os.path.join(dataset_root, video_content)
                else:
                    raise ValueError(f"Video file not found: {video_content}")

        # 情况 3: 异常类型
        else:
            raise TypeError(f"Video path should be str or list, but got {type(video_content)}")
        
        
        if special_process_for_stream:
            # SFT说明：
            # 对于 streaming 样本，会先把原始标注展开成按时间排列的多轮对话。
            # 也就是说，长视频是在这里被转换成 sequential supervision trajectory 的。
            conversation, video_inputs = self.preprocess_conversation_stream(conversation)
            image_inputs = None
        else:
            if not video_inputs and not image_inputs:
                image_inputs, video_inputs = process_vision_info(conversation)

        conversation = [{"role": "previous text", "content": previous_text}] + conversation
        
        texts = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False, return_tensors='pt')
        if return_text: 
            # words_count = len(texts.split())
            # if words_count > 1000:
            #     print(words_count)    
            print(texts)       
            return texts
        
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
        )

        if self.text_sink != 0 or self.text_sliding_window != 0:
            # SFT说明：
            # 这里是在做长文本历史裁剪：
            # 保留前面的固定 sink，再保留最近的一段 sliding window，
            # 中间过长的 previous text 会被裁掉。
            previous_text_start_idx, previous_text_end_idx = self.get_range(inputs.input_ids, 'previous text', 0, contain_lf=True)
            need_truncate = previous_text_start_idx + self.text_sink + self.text_sliding_window <= previous_text_end_idx + 1
            if need_truncate: # 需要裁剪，滑动窗口滑到previous最后一个token
                truncate_start = previous_text_start_idx + self.text_sink
                truncate_end = previous_text_end_idx - self.text_sliding_window
                inputs['input_ids'] = torch.cat([inputs.input_ids[:,:truncate_start],inputs.input_ids[:,truncate_end+1:]],dim=1).contiguous()
                if 'attention_mask' in inputs:
                    inputs['attention_mask'] = torch.cat([inputs.attention_mask[:,:truncate_start],inputs.attention_mask[:,truncate_end+1:]],dim=1).contiguous()

        input_ids = inputs.input_ids
        labels = torch.full_like(input_ids, fill_value=-100, dtype=input_ids.dtype)
        im_start_idxs = (input_ids == self.im_start_id).nonzero()
        im_end_idxs = (input_ids == self.im_end_id).nonzero()
        for (sample_idx, im_start_idx), (sample_idx, im_end_idx) in zip(im_start_idxs, im_end_idxs):
            if input_ids[sample_idx, im_start_idx + 1] == self.assistant_id:
                labels[sample_idx, im_start_idx+3:im_end_idx+1] = input_ids[sample_idx, im_start_idx+3:im_end_idx+1]

        # SFT说明：
        # 只有 assistant span 会参与 NTP loss，user/video prompt token 都会被 mask 掉。
        # 所以当前目标是“预测每一轮 streaming turn 的 assistant 文本”，
        # 而不是“预测显式的 memory action”。
        inputs['labels'] = labels # 构建ntp的label
        if self.return_conversation:
            inputs['conversation'] = conversation
            inputs['start_timestamp'] = conversation[0]['content'][0]['video_start']
        
        return inputs
            


def get_ground_truth(dataset, idx, processor):
    video_text = dataset.getitem(idx, return_text=True)
    ground_truths = []
    for round in video_text:
        if round['role'] == 'assistant':
            ground_truth = round['content'][0]['text']
            ground_truths.append({'ground_truth':ground_truth})
        else:
            continue
    return ground_truths

if __name__ == "__main__":
    import os; os.environ['DATASET_PATH'] = "/workspace/images-ks3-hd/dataset/"
    from collections import defaultdict
    model_path = "/workspace/images-ks3-hd/models/lmm/qwenvl/Qwen2.5-VL-3B-Instruct"
    import argparse
    set_seed(1314)
    args = argparse.ArgumentParser()
    # args.add_argument('--data_path', type=List[str], default=["/workspace/images-ks3-hd/workspace/guanyiran/vid_stream_think_dev/data_gen/debug_yl/curated_sft_streamvlm_format/reasoning_type2_20k_0105_train_with_seeks.jsonl"]) 
    args.add_argument('--data_path', type=List[str], default=["/workspace/images-ks3-hd/workspace/guanyiran/vid_stream_think_dev/data_gen/debug_yl/curated_sft_streamvlm_format/counting_cold_start_01_train_with_seeks.jsonl"]) 
    args.add_argument('--model_base', type=str, default='Qwen',choices=['Qwen'])
    args.add_argument('--idx', type=int, default=0)
    args.add_argument('--text_sink', type=int, default=512)
    args.add_argument('--text_sliding_window', type=int, default=512)
    args = args.parse_args()

    from transformers import AutoProcessor
    
    processor = AutoProcessor.from_pretrained(model_path, padding_side='right') 
    # if args.idx is not None:
    #     model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path,trust_remote_code=True,device_map='auto',attn_implementation='flash_attention_2')
        
    dataset = streamingDataset( # 这里传入了processor
        train_annotation_paths=args.data_path, 
        tokenizer=None,
        processor=processor,
        text_sink=args.text_sink,
        text_sliding_window=args.text_sliding_window,
        with_context=False,
    )
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=dataset.data_collator)
    
    if args.idx is None:
        data = dataset[args.idx]
        # model.generate(input_ids=data.input_ids.to('cuda'),media=data.media,media_config=defaultdict(dict), generation_config=model.default_generation_config)
    else:
        for i, batch in tqdm.tqdm(enumerate(dataloader)):
            import pdb; pdb.set_trace()
            pass
