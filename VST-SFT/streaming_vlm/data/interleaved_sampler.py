import torch
from torch.utils.data import Sampler
from collections import defaultdict, deque

class VideoInterleavedSampler(Sampler[int]):
    def __init__(self, data_source, video_id_field="video_id", generator=None):
        self.data_source = data_source
        self.generator = generator
        
        # 1. 分组：构建 video_id -> deque([idx1, idx2, ...])
        # 使用 deque 是为了后面 pop(0) 操作更高效
        self.video_groups = defaultdict(deque)
        
        print(f"Building index for VideoInterleavedSampler...")
        
        # 快速读取列（针对 HF Dataset 优化）
        if hasattr(data_source, "column_names") and video_id_field in data_source.column_names:
             all_video_ids = data_source[video_id_field]
             for idx, vid in enumerate(all_video_ids):
                 self.video_groups[vid].append(idx)
        else:
            for idx in range(len(data_source)):
                item = data_source[idx]
                vid = item.get(video_id_field) if isinstance(item, dict) else getattr(item, video_id_field)
                self.video_groups[vid].append(idx)

        self.video_keys = list(self.video_groups.keys())
        print(f"Found {len(self.video_keys)} unique videos. Interleaving strategy enabled.")

    def __iter__(self):
        # 2. 随机打乱视频的“处理顺序”
        n_videos = len(self.video_keys)
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        # 随机排列视频 ID
        # 例如: ['Vid_B', 'Vid_A', 'Vid_C']
        perm_indices = torch.randperm(n_videos, generator=generator).tolist()
        shuffled_vids = [self.video_keys[i] for i in perm_indices]

        # 3. 创建临时队列副本，以免修改原始索引
        # 结构: [ (vid_B, deque([0,1])), (vid_A, deque([2,3,4])), ... ]
        active_queues = [
            (vid, list(self.video_groups[vid])) # 转回 list 因为我们要用游标，或者直接 copy deque
            for vid in shuffled_vids
        ]
        
        # 将 list 转换为 iterator，方便逐个获取
        # active_iterators = [iter(indices) for _, indices in active_queues]
        # 上面的写法在视频长度不一致时比较难处理，我们用一种更稳健的轮询方法：
        
        # 使用 deque 存储当前还有剩余数据的视频队列
        # 队列元素: deque([idx1, idx2, ...])
        queues = deque([deque(self.video_groups[vid]) for vid in shuffled_vids])
        
        while queues:
            # 这一轮有多少个视频参与循环
            num_active = len(queues)
            
            for _ in range(num_active):
                # 取出队首的视频队列
                current_video_indices = queues.popleft()
                
                # 吐出该视频的下一个样本 (保持时序)
                yield current_video_indices.popleft()
                
                # 如果该视频还有剩余样本，把它放回队尾，等待下一轮
                if current_video_indices:
                    queues.append(current_video_indices)

    def __len__(self):
        return len(self.data_source)
