

from typing import Optional, Tuple, Union
import torch


from transformers import (
	VisionEncoderDecoderModel,
	GenerationMixin,
)

from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from transformers.models.vision_encoder_decoder.modeling_vision_encoder_decoder import (
	VISION_ENCODER_DECODER_INPUTS_DOCSTRING,
	shift_tokens_right,
)
from torch import nn
from transformers.models.vit.modeling_vit import ViTPatchEmbeddings,ViTEmbeddings,ViTModel
from transformers.utils import  add_start_docstrings_to_model_forward
from .losses import ForCausalLMLoss

class PhuocViTEmbeddings(ViTEmbeddings):
    """
    Construct the CLS token, position and patch embeddings. Optionally, also the mask token.
    """

    def __init__(self, config, use_mask_token: bool = False) -> None:
        super().__init__(config=config,use_mask_token=use_mask_token)

        self.patch_embeddings = ViTPatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = nn.Parameter(torch.randn(1, num_patches + 1, config.hidden_size))
    

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
  
	
	# @classmethod
	# @restore_default_torch_dtype
	# def from_pretrained(
	# 	cls: Type[SpecificPreTrainedModelType],
	# 	pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
	# 	*model_args,
	# 	revision: str = "main",
	# 	weights_only: bool = True,
	# 	commit_hash = None,
	# 	cache_dir: Optional[Union[str, os.PathLike]] = None,
	# 	state_dict = None,
  	# 	use_safetensors =  False,
	# 	ignore_mismatched_sizes: bool = False,
	#     force_download: bool = False,
	#     local_files_only: bool = False,
	#     token: Optional[Union[str, bool]] = None,
	#     config = None,
	# 	**kwargs,
	# ) -> SpecificPreTrainedModelType:
	
	# 	if True:       
	# 		token = None
	# 		from_tf = kwargs.pop("from_tf", False)
	# 		from_flax = kwargs.pop("from_flax", False)
	# 		proxies = kwargs.pop("proxies", None)
	# 		force_download = False
	# 		output_loading_info = kwargs.pop("output_loading_info", False)
	# 		from_pipeline = kwargs.pop("_from_pipeline", None)
	# 		from_auto_class = kwargs.pop("_from_auto", False)
	# 		torch_dtype = kwargs.pop("torch_dtype", None)
	# 		device_map = kwargs.pop("device_map", None)
	# 		max_memory = kwargs.pop("max_memory", None)
	# 		offload_folder = kwargs.pop("offload_folder", None)
	# 		offload_state_dict = kwargs.pop("offload_state_dict", False)
	# 		offload_buffers = kwargs.pop("offload_buffers", False)
	# 		subfolder = kwargs.pop("subfolder", "")
	# 		commit_hash = kwargs.pop("_commit_hash", None)
	# 		variant = kwargs.pop("variant", None)
	# 		adapter_kwargs = kwargs.pop("adapter_kwargs", {})
	# 		adapter_name = kwargs.pop("adapter_name", "default")
	# 		use_flash_attention_2 = kwargs.pop("use_flash_attention_2", False)
	# 		generation_config = kwargs.pop("generation_config", None)
	# 		gguf_file = kwargs.pop("gguf_file", None)
	# 		tp_plan = kwargs.pop("tp_plan", None)
	# 		key_mapping = kwargs.pop("key_mapping", None)
	# 		device_mesh = None
			
	

	# 	# If torchrun was used, make sure to TP by default. This way people don't need to change tp or device map
	# 	if device_map == "auto" and tp_plan is None and int(os.environ.get("WORLD_SIZE", 0)):
	# 		tp_plan = "auto"  # device_map = "auto" in torchrun equivalent to TP plan = AUTO!
	# 		device_map = None

	# 	if commit_hash is None:
	# 		if not isinstance(config, PretrainedConfig):
	# 			# We make a call to the config file first (which may be absent) to get the commit hash as soon as possible
	# 			resolved_config_file = cached_file(
	# 				pretrained_model_name_or_path,
	# 				CONFIG_NAME,
	# 				cache_dir=cache_dir,
	# 				force_download=False,
	# 				proxies=None,
	# 				local_files_only=local_files_only,
	# 				token=None,
	# 				revision=revision,
	# 				subfolder=subfolder,
	# 				_raise_exceptions_for_gated_repo=False,
	# 				_raise_exceptions_for_missing_entries=False,
	# 				_raise_exceptions_for_connection_errors=False,
	# 			)
	# 			commit_hash = extract_commit_hash(resolved_config_file, commit_hash)
	# 		else:
	# 			commit_hash = getattr(config, "_commit_hash", None)	
	# 	_adapter_model_path = None

	# 	# change device_map into a map if we passed an int, a str or a torch.device
	# 	if isinstance(device_map, torch.device):
	# 		device_map = {"": device_map}
	# 	elif isinstance(device_map, str) and device_map not in ["auto", "balanced", "balanced_low_0", "sequential"]:
	# 		try:
	# 			device_map = {"": torch.device(device_map)}
	# 		except RuntimeError:
	# 			raise ValueError(
	# 				"When passing device_map as a string, the value needs to be a device name (e.g. cpu, cuda:0) or "
	# 				f"'auto', 'balanced', 'balanced_low_0', 'sequential' but found {device_map}."
	# 			)
	# 	elif isinstance(device_map, int):
	# 		if device_map < 0:
	# 			raise ValueError(
	# 				"You can't pass device_map as a negative int. If you want to put the model on the cpu, pass device_map = 'cpu' "
	# 			)
	# 		else:
	# 			device_map = {"": device_map}


	# 	user_agent = {"file_type": "model", "framework": "pytorch", "from_auto_class": from_auto_class}
	# 	if from_pipeline is not None:
	# 		user_agent["using_pipeline"] = from_pipeline

	# 	if is_offline_mode() and not local_files_only:
	# 		logger.info("Offline mode: forcing local_files_only=True")
	# 		local_files_only = True

	# 	# Load config if we don't provide a configuration
	# 	if not isinstance(config, PretrainedConfig):
	# 		config_path = config if config is not None else pretrained_model_name_or_path
	# 		config, model_kwargs = cls.config_class.from_pretrained(
	# 			config_path,
	# 			cache_dir=cache_dir,
	# 			return_unused_kwargs=True,
	# 			force_download=force_download,
	# 			proxies=proxies,
	# 			local_files_only=local_files_only,
	# 			token=token,
	# 			revision=revision,
	# 			subfolder=subfolder,
	# 			gguf_file=gguf_file,
	# 			_from_auto=from_auto_class,
	# 			_from_pipeline=from_pipeline,
	# 			**kwargs,
	# 		)
	# 		if "gguf_file" in model_kwargs:
	# 			model_kwargs.pop("gguf_file")
	# 	else:
	# 		# In case one passes a config to `from_pretrained` + "attn_implementation"
	# 		# override the `_attn_implementation` attribute to `attn_implementation` of the kwargs
	# 		# Please see: https://github.com/huggingface/transformers/issues/28038

	# 		# Overwrite `config._attn_implementation` by the one from the kwargs --> in auto-factory
	# 		# we pop attn_implementation from the kwargs but this handles the case where users
	# 		# passes manually the config to `from_pretrained`.
	# 		config = copy.deepcopy(config)

	# 		kwarg_attn_imp = kwargs.pop("attn_implementation", None)
	# 		if kwarg_attn_imp is not None:
	# 			config._attn_implementation = kwarg_attn_imp

	# 		model_kwargs = kwargs

	# 	hf_quantizer = None

		
	# 	checkpoint_files, sharded_metadata = _get_resolved_checkpoint_files(
	# 		pretrained_model_name_or_path=pretrained_model_name_or_path,
	# 		subfolder=subfolder,
	# 		variant=variant,
	# 		gguf_file=gguf_file,
	# 		from_tf=from_tf,
	# 		from_flax=from_flax,
	# 		use_safetensors=use_safetensors,
	# 		cache_dir=cache_dir,
	# 		force_download=force_download,
	# 		proxies=proxies,
	# 		local_files_only=local_files_only,
	# 		token=token,
	# 		user_agent=user_agent,
	# 		revision=revision,
	# 		commit_hash=commit_hash,
	# 	)

		
	# 	is_quantized = hf_quantizer is not None
	# 	from_pt = not (from_tf | from_flax)

	# 	if from_pt:
			
			
	# 		# Find the correct dtype based on current state
	# 		config, torch_dtype, dtype_orig = _get_torch_dtype(
	# 			cls, torch_dtype, checkpoint_files, config, sharded_metadata, state_dict, weights_only,
	# 		)

	# 	config.name_or_path = pretrained_model_name_or_path

	# 	# Instantiate model.
	# 	model_init_context = cls.get_init_context(is_quantized, _is_ds_init_called)

	# 	config = copy.deepcopy(config)  # We do not want to modify the config inplace in from_pretrained.
	# 	if not getattr(config, "_attn_implementation_autoset", False):
	# 		config = cls._autoset_attn_implementation(
	# 			config, use_flash_attention_2=use_flash_attention_2, torch_dtype=torch_dtype, device_map=device_map
	# 		)
		
	# 	with ContextManagers(model_init_context):
	# 		# Let's make sure we don't run the init function of buffer modules
	# 		model = cls(config, *model_args, **model_kwargs)

	# 	# Make sure to tie the weights correctly
	# 	model.tie_weights()

		

	# 	# make sure we use the model's config since the __init__ call might have copied it
	# 	config = model.config

	# 	# Find fp32 modules if needed
	# 	keep_in_fp32_regex = None
	# 	# The _keep_in_fp32_modules flag is only used to avoid bf16 -> fp16 casting precision issues. It was introduced
	# 	# in case of force loading a model that should stay bf16 in fp16 (which includes a few quantizers as this is a pre-processing
	# 	# step for e.g. bitsandbytes). See https://github.com/huggingface/transformers/issues/20287 for details.
	# 	if model._keep_in_fp32_modules is not None and (
	# 		torch_dtype == torch.float16 or getattr(hf_quantizer, "use_keep_in_fp32_modules", False)
	# 	):
	# 		# We need to match exact layers, so we add either `.` on each side, or start/end of string
	# 		keep_in_fp32_regex = re.compile(
	# 			"|".join([rf"((^|\.){module}($|\.))" for module in model._keep_in_fp32_modules])
	# 		)

	
	# 	# Prepare the full device map
	# 	if device_map is not None:
	# 		device_map = _get_device_map(model, device_map, max_memory, hf_quantizer, torch_dtype, keep_in_fp32_regex)

	# 	# Finalize model weight initialization
	# 	if from_tf:
	# 		model, loading_info = cls._load_from_tf(model, config, checkpoint_files)
	# 	elif from_flax:
	# 		model = cls._load_from_flax(model, checkpoint_files)
	# 	elif from_pt:
	# 		# restore default dtype
	# 		if dtype_orig is not None:
	# 			torch.set_default_dtype(dtype_orig)

	# 		(
	# 			model,
	# 			missing_keys,
	# 			unexpected_keys,
	# 			mismatched_keys,
	# 			offload_index,
	# 			error_msgs,
	# 		) = cls._load_pretrained_model(
	# 			model,
	# 			state_dict,
	# 			checkpoint_files,
	# 			pretrained_model_name_or_path,
	# 			ignore_mismatched_sizes=ignore_mismatched_sizes,
	# 			sharded_metadata=sharded_metadata,
	# 			device_map=device_map,
	# 			disk_offload_folder=offload_folder,
	# 			offload_state_dict=offload_state_dict,
	# 			dtype=torch_dtype,
	# 			hf_quantizer=hf_quantizer,
	# 			keep_in_fp32_regex=keep_in_fp32_regex,
	# 			device_mesh=device_mesh,
	# 			key_mapping=key_mapping,
	# 			weights_only=weights_only,
	# 		)

	# 	# make sure token embedding weights are still tied if needed
	# 	model.tie_weights()

	# 	# Set model in evaluation mode to deactivate DropOut modules by default
	# 	model.eval()

	# 	# If it is a model with generation capabilities, attempt to load the generation config
	# 	if model.can_generate() and generation_config is not None:
	# 		logger.info("The user-defined `generation_config` will be used to override the default generation config.")
	# 		model.generation_config = model.generation_config.from_dict(generation_config.to_dict())
	# 	elif model.can_generate() and pretrained_model_name_or_path is not None:
	# 		try:
	# 			model.generation_config = GenerationConfig.from_pretrained(
	# 				pretrained_model_name_or_path,
	# 				cache_dir=cache_dir,
	# 				force_download=force_download,
	# 				proxies=proxies,
	# 				local_files_only=local_files_only,
	# 				token=token,
	# 				revision=revision,
	# 				subfolder=subfolder,
	# 				_from_auto=from_auto_class,
	# 				_from_pipeline=from_pipeline,
	# 				**kwargs,
	# 			)
	# 		except OSError:
	# 			logger.info(
	# 				"Generation config file not found, using a generation config created from the model config."
	# 			)
	# 			pass

	# 	# Dispatch model with hooks on all devices if necessary (not needed with a tp_plan, so we skip it as it slightly
	# 	# harm performances)
	# 	if device_map is not None and device_mesh is None:
	# 		device_map_kwargs = {
	# 			"device_map": device_map,
	# 			"offload_dir": offload_folder,
	# 			"offload_index": offload_index,
	# 			"offload_buffers": offload_buffers,
	# 		}
   
	# 		if "skip_keys" in inspect.signature(dispatch_model).parameters:
	# 			device_map_kwargs["skip_keys"] = model._skip_keys_device_placement
	# 		# For HQQ method we force-set the hooks for single GPU envs
	# 		if (
	# 			"force_hooks" in inspect.signature(dispatch_model).parameters
	# 			and hf_quantizer is not None
	# 			and hf_quantizer.quantization_config.quant_method == QuantizationMethod.HQQ
	# 		):
	# 			device_map_kwargs["force_hooks"] = True
	# 		if (
	# 			hf_quantizer is not None
	# 			and hf_quantizer.quantization_config.quant_method == QuantizationMethod.FBGEMM_FP8
	# 			and isinstance(device_map, dict)
	# 			and ("cpu" in device_map.values() or "disk" in device_map.values())
	# 		):
	# 			device_map_kwargs["offload_buffers"] = True

	# 		if not is_fsdp_enabled() and not is_deepspeed_zero3_enabled():
	# 			dispatch_model(model, **device_map_kwargs)

	
	# 	if _adapter_model_path is not None:
	# 		model.load_adapter(
	# 			_adapter_model_path,
	# 			adapter_name=adapter_name,
	# 			token=token,
	# 			adapter_kwargs=adapter_kwargs,
	# 		)

	# 	if output_loading_info:
	# 		if from_pt:
	# 			loading_info = {
	# 				"missing_keys": missing_keys,
	# 				"unexpected_keys": unexpected_keys,
	# 				"mismatched_keys": mismatched_keys,
	# 				"error_msgs": error_msgs,
	# 			}
	# 		elif from_flax:
	# 			loading_info = None
	# 		return model, loading_info

	# 	return model
