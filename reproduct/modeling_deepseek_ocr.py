from transformers import Qwen3Model, Qwen3ForCausalLM
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.cache_utils import Cache
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

from configuration_deepseek_ocr import DeepseekOCRConfig

from deepencoder_sam import build_sam_vit_b
from deepencoder_clip import build_clip_l

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from addict import Dict
from typing import List, Optional, Tuple, Union

class DeepseekOCRModel(Qwen3Model):
    config_class = DeepseekOCRConfig

    def __init__(self, config: DeepseekOCRConfig):
        super(DeepseekOCRModel, self).__init__(config)

        # 构建 SAM (Segment Anything Model) ViT-B 模型，用于图像分割和特征提取
        self.sam_model = build_sam_vit_b()
        # 构建 CLIP-L 视觉模型，用于图像理解
        self.vision_model = build_clip_l()

        # 投影层设置
        # 创建一个多层感知机投影器，将视觉特征从 2048 维映射到 1280 维，与语言模型的嵌入维度对齐。
        n_embed = 1280
        self.projector =  nn.Linear(2048, n_embed)

        embed_std = 1 / torch.sqrt(torch.tensor(n_embed, dtype=torch.float32))
        # 用于图像tokens的换行标识
        self.image_newline = nn.Parameter(torch.randn(n_embed) * embed_std)
        # 用于区分不同视角(全局/局部)的特征
        self.view_seperator = nn.Parameter(torch.randn(n_embed) * embed_std)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        images: Optional[torch.FloatTensor] = None,
        images_seq_mask: Optional[torch.FloatTensor] = None,
        images_spatial_crop: Optional[torch.FloatTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        
        sam_model = getattr(self, 'sam_model', None)
        vision_model = getattr(self, 'vision_model', None)

        return super(DeepseekOCRModel, self).forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
    
class DeepseekOCRForCausalLM(Qwen3ForCausalLM):
    config_class = DeepseekOCRConfig

    def __init__(self, config):
        super(Qwen3ForCausalLM, self).__init__(config)
        self.model = DeepseekOCRModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        images: Optional[torch.FloatTensor] = None,
        images_seq_mask: Optional[torch.FloatTensor] = None,
        images_spatial_crop: Optional[torch.FloatTensor] = None,
        **kwargs: Unpack[TransformersKwargs],

    ) -> Union[Tuple, CausalLMOutputWithPast]:

        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            images=images,
            images_seq_mask=images_seq_mask,
            images_spatial_crop=images_spatial_crop,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        # logits

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
