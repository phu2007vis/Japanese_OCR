from transformers import (
	VisionEncoderDecoderModel,
	GenerationMixin,
)

from typing import Optional, Tuple, Union
import torch

from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from transformers.models.vision_encoder_decoder.modeling_vision_encoder_decoder import VISION_ENCODER_DECODER_INPUTS_DOCSTRING,_CONFIG_FOR_DOC,shift_tokens_right
from transformers.utils import  add_start_docstrings_to_model_forward, replace_return_docstrings
from losses import ForCausalLMLoss

class PhuocModel(VisionEncoderDecoderModel, GenerationMixin):
	@property
	def loss_function(self):
		return ForCausalLMLoss
	
	@add_start_docstrings_to_model_forward(VISION_ENCODER_DECODER_INPUTS_DOCSTRING)
	# @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
	def forward(
		self,
		pixel_values: Optional[torch.FloatTensor] = None,
		decoder_input_ids: Optional[torch.LongTensor] = None,
		decoder_attention_mask: Optional[torch.BoolTensor] = None,
		encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,
		past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
		decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
		labels: Optional[torch.LongTensor] = None,
		use_cache: Optional[bool] = None,
		output_attentions: Optional[bool] = None,
		output_hidden_states: Optional[bool] = None,
		return_dict: Optional[bool] = None,
		**kwargs,
	) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
	
		return_dict = return_dict if return_dict is not None else self.config.use_return_dict

		# num_items_in_batch is only needed for loss computation
		num_items_in_batch = kwargs.pop("num_items_in_batch", None)

		kwargs_encoder = {argument: value for argument, value in kwargs.items() if not argument.startswith("decoder_")}

		kwargs_decoder = {
			argument[len("decoder_") :]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
		}

		if encoder_outputs is None:
			if pixel_values is None:
				raise ValueError("You have to specify pixel_values")

			encoder_outputs = self.encoder(
				pixel_values=pixel_values,
				output_attentions=output_attentions,
				output_hidden_states=output_hidden_states,
				return_dict=return_dict,
				**kwargs_encoder,
			)
		elif isinstance(encoder_outputs, tuple):
			encoder_outputs = BaseModelOutput(*encoder_outputs)

		encoder_hidden_states = encoder_outputs[0]

		# optionally project encoder_hidden_states
		if (
			self.encoder.config.hidden_size != self.decoder.config.hidden_size
			and self.decoder.config.cross_attention_hidden_size is None
		):
			encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)

		# else:
		encoder_attention_mask = None

		if (labels is not None) and (decoder_input_ids is None and decoder_inputs_embeds is None):
			decoder_input_ids = shift_tokens_right(
				labels, self.config.pad_token_id, self.config.decoder_start_token_id
			)

		# Decode
		decoder_outputs = self.decoder(
			input_ids=decoder_input_ids,
			attention_mask=decoder_attention_mask,
			encoder_hidden_states=encoder_hidden_states,
			encoder_attention_mask=encoder_attention_mask,
			inputs_embeds=decoder_inputs_embeds,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			use_cache=use_cache,
			past_key_values=past_key_values,
			return_dict=return_dict,
			**kwargs_decoder,
		)

		# Compute loss independent from decoder (as some shift the logits inside them)
		loss = None
		if labels is not None:
			logits = decoder_outputs.logits if return_dict else decoder_outputs[0]
			loss = self.loss_function(logits=logits,labels=labels,vocab_size=self.decoder.config.vocab_size,num_items_in_batch=num_items_in_batch)

		if not return_dict:
			if loss is not None:
				return (loss,) + decoder_outputs + encoder_outputs
			else:
				return decoder_outputs + encoder_outputs

		return Seq2SeqLMOutput(
			loss=loss,
			logits=decoder_outputs.logits,
			past_key_values=decoder_outputs.past_key_values,
			decoder_hidden_states=decoder_outputs.hidden_states,
			decoder_attentions=decoder_outputs.attentions,
			cross_attentions=decoder_outputs.cross_attentions,
			encoder_last_hidden_state=encoder_outputs.last_hidden_state,
			encoder_hidden_states=encoder_outputs.hidden_states,
			encoder_attentions=encoder_outputs.attentions,
		)