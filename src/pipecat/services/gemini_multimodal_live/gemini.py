#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import base64
import io
from PIL import Image, UnidentifiedImageError
import json
import time
import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from loguru import logger
from pydantic import BaseModel, Field, model_validator

from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.adapters.services.gemini_adapter import GeminiLLMAdapter
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InputAudioRawFrame,
    InputImageRawFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    LLMSetToolsFrame,
    LLMTextFrame,
    LLMUpdateSettingsFrame,
    StartFrame,
    StartInterruptionFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
    UserImageRawFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.processors.aggregators.llm_response import (
    LLMAssistantAggregatorParams,
    LLMUserAggregatorParams,
)
from pipecat.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
    OpenAILLMContextFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import LLMService
from pipecat.services.openai.llm import (
    OpenAIAssistantContextAggregator,
    OpenAIUserContextAggregator,
)
from pipecat.transcriptions.language import Language
from pipecat.utils.string import match_endofsentence
from pipecat.utils.time import time_now_iso8601

from . import events

try:
    from google import genai
    from google.genai.types import (
        Content,
        LiveConnectConfig,
        SpeechConfig,
        VoiceConfig,
        PrebuiltVoiceConfig,
        Modality,
        Part,
        HttpOptions,
        GenerationConfig,
        Blob,
        ContextWindowCompressionConfig,
        RealtimeInputConfig,
        AutomaticActivityDetection,
        Tool,
        AudioTranscriptionConfig,
        SessionResumptionConfig,
        SlidingWindow,  # For ContextWindowCompressionParams -> SDK type
    )
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use the Google GenAI SDK, you need to `pip install google-genai`."
    )
    raise Exception(
        f"Missing module: {e}. Please install google-genai."
    ) from e


def language_to_gemini_language(language: Language) -> Optional[str]:
    """Maps a Language enum value to a Gemini Live supported language code.

    Source:
    https://ai.google.dev/api/generate-content#MediaResolution

    Returns None if the language is not supported by Gemini Live.
    """
    language_map = {
        # Arabic
        Language.AR: "ar-XA",
        # Bengali
        Language.BN_IN: "bn-IN",
        # Chinese (Mandarin)
        Language.CMN: "cmn-CN",
        Language.CMN_CN: "cmn-CN",
        Language.ZH: "cmn-CN",  # Map general Chinese to Mandarin for Gemini
        Language.ZH_CN: "cmn-CN",  # Map Simplified Chinese to Mandarin for Gemini
        # German
        Language.DE: "de-DE",
        Language.DE_DE: "de-DE",
        # English
        # Default to US English (though not explicitly listed in supported codes)
        Language.EN: "en-US",
        Language.EN_US: "en-US",
        Language.EN_AU: "en-AU",
        Language.EN_GB: "en-GB",
        Language.EN_IN: "en-IN",
        # Spanish
        Language.ES: "es-ES",  # Default to Spain Spanish
        Language.ES_ES: "es-ES",
        Language.ES_US: "es-US",
        # French
        Language.FR: "fr-FR",  # Default to France French
        Language.FR_FR: "fr-FR",
        Language.FR_CA: "fr-CA",
        # Gujarati
        Language.GU: "gu-IN",
        Language.GU_IN: "gu-IN",
        # Hindi
        Language.HI: "hi-IN",
        Language.HI_IN: "hi-IN",
        # Indonesian
        Language.ID: "id-ID",
        Language.ID_ID: "id-ID",
        # Italian
        Language.IT: "it-IT",
        Language.IT_IT: "it-IT",
        # Japanese
        Language.JA: "ja-JP",
        Language.JA_JP: "ja-JP",
        # Kannada
        Language.KN: "kn-IN",
        Language.KN_IN: "kn-IN",
        # Korean
        Language.KO: "ko-KR",
        Language.KO_KR: "ko-KR",
        # Malayalam
        Language.ML: "ml-IN",
        Language.ML_IN: "ml-IN",
        # Marathi
        Language.MR: "mr-IN",
        Language.MR_IN: "mr-IN",
        # Dutch
        Language.NL: "nl-NL",
        Language.NL_NL: "nl-NL",
        # Polish
        Language.PL: "pl-PL",
        Language.PL_PL: "pl-PL",
        # Portuguese (Brazil)
        Language.PT_BR: "pt-BR",
        # Russian
        Language.RU: "ru-RU",
        Language.RU_RU: "ru-RU",
        # Tamil
        Language.TA: "ta-IN",
        Language.TA_IN: "ta-IN",
        # Telugu
        Language.TE: "te-IN",
        Language.TE_IN: "te-IN",
        # Thai
        Language.TH: "th-TH",
        Language.TH_TH: "th-TH",
        # Turkish
        Language.TR: "tr-TR",
        Language.TR_TR: "tr-TR",
        # Vietnamese
        Language.VI: "vi-VN",
        Language.VI_VN: "vi-VN",
    }
    return language_map.get(language)


class GeminiMultimodalLiveContext(OpenAILLMContext):
    @staticmethod
    def upgrade(obj: OpenAILLMContext) -> "GeminiMultimodalLiveContext":
        if isinstance(obj, OpenAILLMContext) and not isinstance(obj, GeminiMultimodalLiveContext):
            logger.debug(f"Upgrading to Gemini Multimodal Live Context: {obj}")
            obj.__class__ = GeminiMultimodalLiveContext
            obj._restructure_from_openai_messages()
        return obj

    def _restructure_from_openai_messages(self):
        pass

    def extract_system_instructions(self):
        system_instruction = ""
        for item in self.messages:
            if item.get("role") == "system":
                content = item.get("content", "")
                if content:
                    if system_instruction and not system_instruction.endswith("\n"):
                        system_instruction += "\n"
                    system_instruction += str(content)
        return system_instruction

    def get_messages_for_initializing_history(self):
        messages = []
        for item in self.messages:
            role = item.get("role")

            if role == "system":
                continue

            elif role == "assistant":
                role = "model"

            content = item.get("content")
            parts = []
            if isinstance(content, str):
                parts = [{"text": content}]
            elif isinstance(content, list):
                for part in content:
                    if part.get("type") == "text":
                        parts.append({"text": part.get("text")})
                    else:
                        logger.warning(
                            f"Unsupported content type: {str(part)[:80]}")
            else:
                logger.warning(
                    f"Unsupported content type: {str(content)[:80]}")
            messages.append({"role": role, "parts": parts})
        return messages


class GeminiMultimodalLiveUserContextAggregator(OpenAIUserContextAggregator):
    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        # kind of a hack just to pass the LLMMessagesAppendFrame through, but it's fine for now
        if isinstance(frame, LLMMessagesAppendFrame):
            await self.push_frame(frame, direction)


class GeminiMultimodalLiveAssistantContextAggregator(OpenAIAssistantContextAggregator):
    # The LLMAssistantContextAggregator uses TextFrames to aggregate the LLM output,
    # but the GeminiMultimodalLiveAssistantContextAggregator pushes LLMTextFrames and TTSTextFrames. We
    # need to override this proces_frame for LLMTextFrame, so that only the TTSTextFrames
    # are process. This ensures that the context gets only one set of messages.
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if not isinstance(frame, LLMTextFrame):
            await super().process_frame(frame, direction)

    async def handle_user_image_frame(self, frame: UserImageRawFrame):
        # We don't want to store any images in the context. Revisit this later
        # when the API evolves.
        pass


@dataclass
class GeminiMultimodalLiveContextAggregatorPair:
    _user: GeminiMultimodalLiveUserContextAggregator
    _assistant: GeminiMultimodalLiveAssistantContextAggregator

    def user(self) -> GeminiMultimodalLiveUserContextAggregator:
        return self._user

    def assistant(self) -> GeminiMultimodalLiveAssistantContextAggregator:
        return self._assistant


class GeminiMultimodalModalities(Enum):
    TEXT = "TEXT"
    AUDIO = "AUDIO"


class GeminiMediaResolution(str, Enum):
    """Media resolution options for Gemini Multimodal Live."""

    UNSPECIFIED = "MEDIA_RESOLUTION_UNSPECIFIED"  # Use default
    LOW = "MEDIA_RESOLUTION_LOW"  # 64 tokens
    MEDIUM = "MEDIA_RESOLUTION_MEDIUM"  # 256 tokens
    HIGH = "MEDIA_RESOLUTION_HIGH"  # Zoomed reframing with 256 tokens


# Note: GeminiVADParams is deprecated. Use RealtimeInputConfig in InputParams instead.
class GeminiVADParams(BaseModel):
    """Voice Activity Detection parameters. 

    DEPRECATED: Use RealtimeInputConfig in InputParams instead.
    This class is kept for backward compatibility but is no longer used internally.
    """

    disabled: Optional[bool] = Field(default=None)
    start_sensitivity: Optional[events.StartSensitivity] = Field(default=None)
    end_sensitivity: Optional[events.EndSensitivity] = Field(default=None)
    prefix_padding_ms: Optional[int] = Field(default=None)
    silence_duration_ms: Optional[int] = Field(default=None)


class SlidingWindowParams(BaseModel):
    target_tokens: int


class ContextWindowCompressionParams(BaseModel):
    """Parameters for context window compression."""

    # We\'ll keep this to control if the SDK object is created
    enabled: bool = Field(default=False)
    trigger_tokens: Optional[int] = Field(default=None)
    sliding_window: Optional[SlidingWindowParams] = Field(default=None)


class InputParams(BaseModel):
    """Parameters to configure the Gemini Live connection, mirroring LiveConnectConfig."""

    context_window_compression: Optional[ContextWindowCompressionParams] = Field(
        default=None)
    # enable_affective_dialog: Optional[bool] = Field(default=None)  # Not available in current SDK
    generation_config: Optional[GenerationConfig] = Field(default=None)
    # http_options: Optional[HttpOptions] = Field(default=None)  # Not available in current SDK LiveConnectConfig
    input_audio_transcription: Optional[AudioTranscriptionConfig] = Field(
        default=None)
    max_output_tokens: Optional[int] = Field(default=None)
    media_resolution: Optional[GeminiMediaResolution] = Field(default=None)
    output_audio_transcription: Optional[AudioTranscriptionConfig] = Field(
        default=None)
    # proactivity: Optional[ProactivityConfig] = Field(default=None)  # Not available in current SDK
    realtime_input_config: Optional[RealtimeInputConfig] = Field(default=None)
    response_modalities: Optional[List[Modality]] = Field(default=None)
    seed: Optional[int] = Field(default=None)
    session_resumption: Optional[SessionResumptionConfig] = Field(default=None)
    speech_config: Optional[SpeechConfig] = Field(default=None)

    temperature: Optional[float] = Field(default=None)
    top_k: Optional[float] = Field(default=None)
    top_p: Optional[float] = Field(default=None)

    extra: Optional[Dict[str, Any]] = Field(default_factory=dict)


class GeminiMultimodalLiveLLMService(LLMService):
    """Provides access to Google's Gemini Multimodal Live API.

    This service enables real-time conversations with Gemini, supporting both
    text and audio modalities. It handles voice transcription, streaming audio
    responses, and tool usage.

    Args:
        api_key (str): Google AI API key
        base_url (str, optional): API endpoint base URL. Defaults to
            "generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent".
        model (str, optional): Model identifier to use. Defaults to
            "models/gemini-2.0-flash-live-001".
        voice_id (str, optional): TTS voice identifier. Defaults to "Charon".
        start_audio_paused (bool, optional): Whether to start with audio input paused.
            Defaults to False.
        start_video_paused (bool, optional): Whether to start with video input paused.
            Defaults to False.
        system_instruction (str, optional): System prompt for the model. Defaults to None.
        tools (Union[List[dict], ToolsSchema], optional): Tools/functions available to the model.
            Defaults to None.
        params (InputParams, optional): Configuration parameters for the model.
            Defaults to InputParams().
        inference_on_context_initialization (bool, optional): Whether to generate a response
            when context is first set. Defaults to True.
    """

    # Overriding the default adapter to use the Gemini one.
    adapter_class = GeminiLLMAdapter

    def __init__(
        self,
        *,
        api_key: str,
        # base_url is removed
        # Updated model to match reference
        model: str = "models/gemini-2.0-flash-live-001",
        voice_id: str = "Charon",  # Example: "Aoede", "Puck", etc.
        language: Language = Language.EN_US,  # Default service language
        start_audio_paused: bool = False,
        # Video not explicitly in scope yet, but keep for now
        start_video_paused: bool = False,
        system_instruction: Optional[str] = None,
        # Tool handling will need to map to SDK
        tools: Optional[Union[List[dict], ToolsSchema]] = None,
        # Will be used to construct LiveConnectConfig
        params: Optional[InputParams] = None,
        inference_on_context_initialization: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)  # Removed base_url

        params = params or InputParams()

        # Set default response modalities for audio-to-audio applications
        # if none are explicitly provided
        if params.response_modalities is None:
            params.response_modalities = [Modality.AUDIO]
            logger.debug(
                "Setting default response_modalities to [Modality.AUDIO] for audio-to-audio functionality")

        self._input_params = params

        self._default_voice_id = voice_id
        self._default_language_code = language_to_gemini_language(
            language) or "en-US"

        # Initialize Google GenAI Client
        # Note: http_options not available in current SDK's LiveConnectConfig
        # but can still be used for client initialization
        client_options = HttpOptions(api_version="v1beta")
        self._client = genai.Client(
            api_key=api_key, http_options=client_options)
        self._session = None  # Will hold the live session
        # Will hold the async context manager for the session
        self._session_connector_obj = None

        # Still useful for potential re-connections or other direct API calls if any
        self._api_key = api_key
        self.set_model_name(model)  # Uses self._model_name
        self._voice_id = voice_id

        # Will be used in LiveConnectConfig or sent
        if system_instruction and isinstance(system_instruction, str):
            self._system_instruction = Content(
                parts=[Part(text=system_instruction)])
        elif system_instruction:  # Assuming it\'s already a Content object if not a string
            self._system_instruction = system_instruction
        else:
            self._system_instruction = None
        self._tools = tools  # Tool handling to be adapted
        self._inference_on_context_initialization = inference_on_context_initialization
        # May still be relevant for managing turns
        self._needs_turn_complete_message = False

        self._audio_input_paused = start_audio_paused
        self._video_input_paused = start_video_paused  # Video params for future
        # Keep context management
        self._context: Optional[GeminiMultimodalLiveContext] = None

        # Initialize settings dict for dynamic updates
        self._settings = {}

        # Remove WebSocket specific state
        # self._websocket = None
        # self._receive_task = None
        # self._disconnecting = False
        # self._api_session_ready = False
        # self._run_llm_when_api_session_ready = False

        self._user_is_speaking = False
        self._bot_is_speaking = False
        self._user_audio_buffer = bytearray()
        self._user_transcription_buffer = ""
        self._last_transcription_sent = ""  # May be useful for debouncing/diffing
        self._bot_audio_buffer = bytearray()
        self._bot_text_buffer = ""

        # Audio sample rates:
        # Input to Gemini Live API: Raw 16 bit PCM audio at 16kHz (mono)
        # Output from Gemini Live API: Raw 16 bit PCM audio at 24kHz (mono)
        # The current _sample_rate = 24000 seems to be for output.
        # We'll need to ensure input audio is handled at 16kHz.
        self._output_audio_sample_rate = 24000
        self._input_audio_sample_rate = 16000  # Define for clarity

        # Store relevant params from InputParams for LiveConnectConfig construction
        # These will be used when establishing the session, not stored in a _settings dict
        self._input_params = params
        self._llm_task = None
        self._outgoing_sdk_queue = asyncio.Queue()

        self._session_closed_flag = False
        # Configuration will be done via LiveConnectConfig during the connect call.

    def can_generate_metrics(self) -> bool:
        return True

    def set_audio_input_paused(self, paused: bool):
        self._audio_input_paused = paused

    def set_video_input_paused(self, paused: bool):
        self._video_input_paused = paused

    def set_model_modalities(self, modalities: GeminiMultimodalModalities):
        """Set the response modalities for the model."""
        if modalities == GeminiMultimodalModalities.TEXT:
            self._input_params.response_modalities = [Modality.TEXT]
        elif modalities == GeminiMultimodalModalities.AUDIO:
            self._input_params.response_modalities = [Modality.AUDIO]
        else:
            # Default to both
            self._input_params.response_modalities = [
                Modality.TEXT, Modality.AUDIO]

        # Store in settings for potential future use
        self._settings["modalities"] = modalities

    def set_language(self, language: Language):
        """Set the language for generation."""
        self._language = language
        self._language_code = language_to_gemini_language(language) or "en-US"

        # Update the default language code for future speech configs
        self._default_language_code = self._language_code

        # If we don't have a custom speech_config, this will be used in _establish_sdk_session
        # If we do have a custom one, the user should update it manually
        self._settings["language"] = self._language_code
        logger.info(f"Set Gemini language to: {self._language_code}")

    async def set_context(self, context: OpenAILLMContext):
        """Set the context explicitly from outside the pipeline.

        This is useful when initializing a conversation because in server-side VAD mode we might not have a
        way to trigger the pipeline. This sends the history to the server. The `inference_on_context_initialization`
        flag controls whether to set the turnComplete flag when we do this. Without that flag, the model will
        not respond. This is often what we want when setting the context at the beginning of a conversation.
        """
        if self._context:
            logger.error(
                "Context already set. Can only set up Gemini Multimodal Live context once."
            )
            return
        self._context = GeminiMultimodalLiveContext.upgrade(context)
        await self._create_initial_response()

    async def _update_settings(self, settings=None):
        """Update model settings. For now, this is a placeholder since the SDK session
        needs to be recreated to change most settings."""
        if settings:
            self._settings.update(settings)

        # Note: Most LiveConnectConfig settings require recreating the session
        # For now, we just store the settings for potential future use
        logger.debug(f"Settings updated: {self._settings}")

        # TODO: Implement dynamic settings updates if the SDK supports it in the future
        # For now, settings changes would require reconnection

    #
    # standard AIService frame handling
    #

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self._establish_sdk_session()

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._disconnect()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if not self._session and not isinstance(frame, (StartFrame, EndFrame, CancelFrame, OpenAILLMContextFrame)):
            logger.warning(
                f"Gemini SDK session not active, cannot process {type(frame)}. Frame will be dropped or pushed if not data.")
            # Allow non-data frames to pass for pipeline mechanics if necessary
            if not isinstance(frame, (LLMMessagesAppendFrame, InputAudioRawFrame, UserImageRawFrame)):
                await self.push_frame(frame, direction)
            return

        if isinstance(frame, StartFrame):
            if not self._llm_task or self._llm_task.done():
                logger.debug(
                    f"Gemini Service: StartFrame received. self._session is {'set' if self._session else 'None'}, self._llm_task is {'set' if self._llm_task else 'None'}, task_done: {self._llm_task.done() if self._llm_task else 'N/A'}")
                if self._session:
                    logger.info(
                        "Starting SDK send/receive loop from StartFrame.")
                    self._llm_task = self.create_task(
                        self._sdk_send_receive_loop())
                else:
                    logger.error(
                        "Session not established on StartFrame, cannot start LLM task.")
                    await self.push_frame(ErrorFrame("Gemini session failed to start."))
            await self.push_frame(frame, direction)
        elif isinstance(frame, (EndFrame, CancelFrame)):
            await self.push_frame(frame, direction)
        elif isinstance(frame, OpenAILLMContextFrame):
            logger.debug(f"Processing OpenAILLMContextFrame.")
            self._context = GeminiMultimodalLiveContext.upgrade(frame.context)
            if hasattr(frame.context, 'tools') and frame.context.tools:
                self._tools = frame.context.tools

            if self._session and (not self._llm_task or self._llm_task.done()):
                logger.info(
                    "Starting SDK send/receive loop from OpenAILLMContextFrame.")
                self._llm_task = self.create_task(
                    self._sdk_send_receive_loop())
            await self.push_frame(frame, direction)

        elif isinstance(frame, LLMMessagesAppendFrame):
            logger.debug(
                f"Queueing LLMMessagesAppendFrame for SDK: {frame.messages}")
            turns = []
            for msg in frame.messages:
                role = "model" if msg.get(
                    "role") == "assistant" else msg.get("role", "user")
                content = msg.get("content")
                parts = []
                if isinstance(content, str):
                    parts.append(Part(text=content))
                elif isinstance(content, list):
                    for item_part in content:
                        if item_part.get("type") == "text":
                            parts.append(Part(text=item_part.get("text")))
                if parts:
                    turns.append(Content(role=role, parts=parts))
            if turns:
                await self._outgoing_sdk_queue.put({"turns": turns})

        elif isinstance(frame, InputAudioRawFrame):
            if self._audio_input_paused:
                await self.push_frame(frame, direction)
                return

            audio_bytes = frame.audio
            if frame.sample_rate != self._input_audio_sample_rate:
                logger.warning(
                    f"Input audio SR {frame.sample_rate} != expected {self._input_audio_sample_rate}. Resampling needed (not implemented). Sending as is.")

            # Using Blob for inline_data as per google.generativeai.types.Part documentation
            audio_part = Part(inline_data=Blob(
                mime_type=f'audio/l16;rate={self._input_audio_sample_rate}', data=audio_bytes))
            await self._outgoing_sdk_queue.put({"turns": [Content(parts=[audio_part])]})
            # Also push the original frame for other processors like VAD
            await self.push_frame(frame, direction)

        elif isinstance(frame, UserImageRawFrame):
            logger.warning(
                "Video input frame received, but not yet supported in this refactoring.")
            await self.push_frame(frame, direction)

        elif isinstance(frame, StartInterruptionFrame):
            if self._bot_is_speaking:
                self._bot_is_speaking = False
                await self.push_frame(TTSStoppedFrame())
            if self._bot_text_buffer:
                await self.push_frame(LLMFullResponseEndFrame())
            self._bot_text_buffer = ""
            self._bot_audio_buffer = bytearray()
            # TODO: Consider if/how to signal interruption to the SDK if it supports it.
            # For now, this is local state clearing and frame emission.
            logger.debug("Interruption frame handled locally.")
            await self.push_frame(frame, direction)

        elif isinstance(frame, (UserStartedSpeakingFrame, UserStoppedSpeakingFrame,
                                BotStartedSpeakingFrame, BotStoppedSpeakingFrame)):
            if isinstance(frame, UserStartedSpeakingFrame):
                self._user_is_speaking = True
            if isinstance(frame, UserStoppedSpeakingFrame):
                self._user_is_speaking = False
            await self.push_frame(frame, direction)

        elif isinstance(frame, LLMSetToolsFrame):
            logger.info(f"LLMSetToolsFrame received. Tools: {frame.tools}")
            self._tools = frame.tools
            logger.warning(
                "LLMSetToolsFrame: Reconnecting with new tools not yet implemented. Tools will apply on next full session start.")
            await self.push_frame(frame, direction)

        elif isinstance(frame, LLMUpdateSettingsFrame):
            logger.info(
                f"LLMUpdateSettingsFrame received. Settings: {frame.settings}")
            # TODO: Implement mapping from frame.settings to self._input_params and handle session update/restart.
            logger.warning(
                "LLMUpdateSettingsFrame: Updating settings mid-session not fully implemented. Changes may require session restart.")
            await self.push_frame(frame, direction)

        else:
            await self.push_frame(frame, direction)

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._disconnect()

    #
    # usage metadata
    #
    async def _handle_usage_metadata(self, usage_metadata):
        # TODO: Process and potentially push usage metadata if relevant for pipecat metrics
        logger.debug(f"Gemini SDK Usage Metadata: {usage_metadata}")
        # Example: if hasattr(usage_metadata, 'prompt_token_count') and usage_metadata.prompt_token_count:
        #    self.llm_token_usage.update_prompt_tokens(usage_metadata.prompt_token_count)
        # if hasattr(usage_metadata, 'response_token_count') and usage_metadata.response_token_count:
        #    self.llm_token_usage.update_completion_tokens(usage_metadata.response_token_count)
        pass

    #
    # speech and interruption handling
    #

    async def _handle_interruption(self):
        self._bot_is_speaking = False
        await self.push_frame(TTSStoppedFrame())
        await self.push_frame(LLMFullResponseEndFrame())

    async def _handle_user_started_speaking(self, frame):
        self._user_is_speaking = True
        pass

    async def _handle_user_stopped_speaking(self, frame):
        self._user_is_speaking = False
        self._user_audio_buffer = bytearray()
        if self._needs_turn_complete_message:
            self._needs_turn_complete_message = False
            evt = events.ClientContentMessage.model_validate(
                {"clientContent": {"turnComplete": True}}
            )
            await self.send_client_event(evt)

    #
    # frame processing
    #
    # StartFrame, StopFrame, CancelFrame implemented in base class
    #

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            await self.push_frame(frame, direction)
        elif isinstance(frame, OpenAILLMContextFrame):
            context: GeminiMultimodalLiveContext = GeminiMultimodalLiveContext.upgrade(
                frame.context
            )
            # For now, we'll only trigger inference here when either:
            #   1. We have not seen a context frame before
            #   2. The last message is a tool call result
            if not self._context:
                self._context = context
                if frame.context.tools:
                    self._tools = frame.context.tools
                await self._create_initial_response()
            elif context.messages and context.messages[-1].get("role") == "tool":
                # Support just one tool call per context frame for now
                tool_result_message = context.messages[-1]
                await self._tool_result(tool_result_message)
        elif isinstance(frame, InputAudioRawFrame):
            await self._send_user_audio(frame)
            await self.push_frame(frame, direction)
        elif isinstance(frame, InputImageRawFrame):
            await self._send_user_video(frame)
            await self.push_frame(frame, direction)
        elif isinstance(frame, StartInterruptionFrame):
            await self._handle_interruption()
            await self.push_frame(frame, direction)
        elif isinstance(frame, UserStartedSpeakingFrame):
            await self._handle_user_started_speaking(frame)
            await self.push_frame(frame, direction)
        elif isinstance(frame, UserStoppedSpeakingFrame):
            await self._handle_user_stopped_speaking(frame)
            await self.push_frame(frame, direction)
        elif isinstance(frame, BotStartedSpeakingFrame):
            # Ignore this frame. Use the serverContent API message instead
            await self.push_frame(frame, direction)
        elif isinstance(frame, BotStoppedSpeakingFrame):
            # ignore this frame. Use the serverContent.turnComplete API message
            await self.push_frame(frame, direction)
        elif isinstance(frame, LLMMessagesAppendFrame):
            await self._create_single_response(frame.messages)
        elif isinstance(frame, LLMUpdateSettingsFrame):
            await self._update_settings(frame.settings)
        elif isinstance(frame, LLMSetToolsFrame):
            await self._update_settings()
        else:
            await self.push_frame(frame, direction)

    #
    # websocket communication
    #

    async def _send_user_audio(self, frame: InputAudioRawFrame):
        if self._audio_input_paused:
            logger.debug("Audio input is paused, not sending audio frame.")
            return

        # TODO: Ensure audio is 16kHz, 16-bit PCM, mono.
        # This might involve resampling if frame.sample_rate != self._input_audio_sample_rate (16000)
        # or channel conversion if frame.num_channels != 1.
        # For now, we assume the audio in frame.audio is compliant or this is handled upstream.
        # Also ensure frame.audio is bytes.
        if not (hasattr(frame, 'audio') and isinstance(frame.audio, bytes) and frame.audio):
            logger.warning(
                "InputAudioRawFrame is missing audio data or data is not bytes. Cannot send.")
            return

        audio_data = frame.audio

        # The reference_live_api.py uses "audio/pcm" for session.send(input=...).
        # The Gemini docs specify "Raw 16 bit PCM audio at 16kHz little-endian".
        # Let's use a more specific mime type if possible, but ensure it works with session.send().
        # The reference uses "audio/pcm" and sets sample rate etc. on the PyAudio stream.
        # For session.send(), the data itself is key.
        # The reference example for session.send() with audio:
        # await self.out_queue.put({"data": data, "mime_type": "audio/pcm"})
        # And then: await self.session.send(input=msg)
        # So, the payload for session.send should be a dict like:
        # {"data": audio_bytes, "mime_type": "audio/pcm"}
        # The sample rate (16kHz) and format (16-bit PCM mono) are implicit expectations for the data bytes.

        # Using "audio/pcm" as per reference for session.send()
        # The actual format (16kHz, 16-bit, mono) must be ensured by the input frame.audio.
        mime_type = "audio/pcm"
        # If a more specific mime_type like "audio/L16;rate=16000;channels=1" is required by the API
        # for session.send(input=...), this might need adjustment.
        # For now, aligning with the reference's session.send() usage.

        try:
            audio_payload = {
                "data": audio_data,
                "mime_type": mime_type  # Using "audio/pcm"
            }

            sdk_message = {"type": "media_input", "payload": audio_payload}

            if self._session and self._outgoing_sdk_queue:
                await self._outgoing_sdk_queue.put(sdk_message)
            else:
                logger.warning(
                    "SDK session or outgoing queue not available. Cannot send audio frame.")

        except Exception as e:
            logger.opt(exception=e).error(
                "Error processing or queuing audio frame.")
            await self.push_frame(ErrorFrame(f"Error processing audio frame: {e}"))

        # Retain local audio buffering logic from old implementation for pipecat's VAD/transcription.
        if hasattr(self, "_user_audio_buffer"):
            audio_to_buffer = frame.audio  # Use the original frame audio for local buffer
            if self._user_is_speaking:
                self._user_audio_buffer.extend(audio_to_buffer)
            else:
                self._user_audio_buffer.extend(audio_to_buffer)
                # Calculate length for 0.5 seconds of audio (16-bit PCM = 2 bytes per sample)
                try:
                    length_in_bytes = int(
                        (frame.sample_rate * frame.num_channels * 2) * 0.5)
                    if length_in_bytes > 0:
                        self._user_audio_buffer = self._user_audio_buffer[-length_in_bytes:]
                    else:  # frame.sample_rate or frame.num_channels might be 0
                        self._user_audio_buffer = bytearray()
                except Exception as e:  # Catch any error during calculation
                    logger.opt(exception=e).warning(
                        "Error calculating audio buffer length.")
                    self._user_audio_buffer = bytearray()  # Reset buffer on error
        else:
            logger.warning(
                "_user_audio_buffer attribute not found, skipping local audio buffering.")

    async def _create_initial_response(self):
        if not self._session:
            logger.warning(
                "SDK session not ready, cannot create initial response yet. Will attempt when session is ready.")
            # This matches the old logic of self._run_llm_when_api_session_ready = True
            # The calling logic should handle retrying or calling this when session is ready.
            # For example, _establish_sdk_session could set a flag or call this upon success.
            return

        if not self._context:
            logger.warning(
                "Context not available, cannot create initial response.")
            return

        initial_turns_data = self._context.get_messages_for_initializing_history()
        if not initial_turns_data:
            logger.debug("No initial messages in context to send.")
            return

        logger.debug(
            f"Preparing to send initial context/history ({len(initial_turns_data)} turns)")

        try:
            for turn_data in initial_turns_data:
                role = turn_data.get("role")
                parts_data = turn_data.get("parts")

                if not role or not parts_data:
                    logger.warning(
                        f"Skipping invalid turn data for initial response: {turn_data}")
                    continue

                sdk_parts = []
                for part_item in parts_data:
                    if "text" in part_item:
                        sdk_parts.append(Part(text=part_item["text"]))
                    # TODO: Handle other part types like Blob if necessary for initial context

                if not sdk_parts:
                    logger.warning(
                        f"Skipping initial turn with no valid parts: {turn_data}")
                    continue

                content_to_send = Content(role=role, parts=sdk_parts)

                if self._outgoing_sdk_queue:
                    await self._outgoing_sdk_queue.put(content_to_send)
                    logger.debug(
                        f"Queued initial message: Role {role}, Parts: {len(sdk_parts)}")
                else:
                    # This should not happen if self._session is valid, as queue is initialized with session
                    logger.error(
                        "Outgoing SDK queue not available for initial response despite session being ready.")
                    break  # Stop trying to send further initial messages

            # Regarding self._inference_on_context_initialization and self._needs_turn_complete_message:
            # The SDK's send_client_content implicitly handles turns. If _inference_on_context_initialization is true,
            # the server will infer the turn completion based on the content received.
            # These flags might be less relevant or need different handling with the SDK.
            # The original _needs_turn_complete_message was for a specific websocket message.
            logger.debug("Finished queueing initial responses.")

        except Exception as e:
            logger.opt(exception=e).error(
                "Error preparing or queuing initial response.")
            await self.push_frame(ErrorFrame(f"Error sending initial response: {e}"))

    async def _send_user_video(self, frame: InputImageRawFrame):
        if self._video_input_paused:
            logger.debug("Video input is paused, not sending video frame.")
            return

        if not (hasattr(frame, 'image') and frame.image and
                hasattr(frame, 'format') and frame.format and
                hasattr(frame, 'size') and frame.size and
                isinstance(frame.size, tuple) and len(frame.size) == 2):
            logger.warning(
                "InputImageRawFrame is missing image data, format, or valid size. Cannot send.")
            return

        try:
            logger.debug(
                f"Processing InputImageRawFrame. Original format: {frame.format}, Size: {frame.size}, Image bytes length: {len(frame.image)}")

            img: Image.Image  # Type hint for PIL Image object

            if frame.format.upper() == "RGB":
                # Ensure image data length matches width * height * 3 (for RGB)
                expected_len = frame.size[0] * frame.size[1] * 3
                if len(frame.image) != expected_len:
                    logger.error(
                        f"RGB image data length ({len(frame.image)}) does not match expected length ({expected_len}) for size {frame.size}. Cannot process.")
                    await self.push_frame(ErrorFrame(f"Corrupt RGB image data for size {frame.size}"))
                    return
                img = Image.frombytes("RGB", frame.size, frame.image)
            # Add other common formats if needed
            elif frame.format.upper() in ["JPEG", "JPG", "PNG"]:
                img_byte_arr = io.BytesIO(frame.image)
                img = Image.open(img_byte_arr)
            else:
                # Attempt to open with PIL, hoping it can identify it
                # This path might be risky if format is unknown and not a standard file
                logger.warning(
                    f"Unknown image format '{frame.format}'. Attempting to open with PIL.Image.open().")
                try:
                    img_byte_arr = io.BytesIO(frame.image)
                    img = Image.open(img_byte_arr)
                except Exception as pil_e:
                    logger.error(
                        f"PIL.Image.open() failed for format '{frame.format}': {pil_e}")
                    await self.push_frame(ErrorFrame(f"Cannot identify/process image format: {frame.format}"))
                    return

            # Convert to RGB if it's not (e.g. RGBA, P) for consistency before saving to JPEG
            if img.mode != "RGB":
                img = img.convert("RGB")

            jpeg_image_io = io.BytesIO()
            img.save(jpeg_image_io, format="JPEG", quality=85)
            jpeg_image_bytes = jpeg_image_io.getvalue()

            base64_encoded_image = base64.b64encode(
                jpeg_image_bytes).decode('utf-8')

            image_payload = {
                "mime_type": "image/jpeg",  # Always sending JPEG to Gemini
                "data": base64_encoded_image
            }

            sdk_message = {"type": "media_input", "payload": image_payload}

            if self._session and self._outgoing_sdk_queue:
                await self._outgoing_sdk_queue.put(sdk_message)
            else:
                logger.warning(
                    "SDK session or outgoing queue not available. Cannot send processed video frame.")

        except UnidentifiedImageError as pil_e:  # Catch specific PIL error
            logger.opt(exception=pil_e).error(
                f"PIL UnidentifiedImageError processing video frame (format: {frame.format}).")
            await self.push_frame(ErrorFrame(f"PIL cannot identify image (format {frame.format}): {pil_e}"))
        except Exception as e:
            logger.opt(exception=e).error(
                "Error processing or queuing video frame.")
            await self.push_frame(ErrorFrame(f"Error processing video frame: {e}"))

    async def _establish_sdk_session(self):
        if self._session:
            logger.debug("SDK session already established.")
            return

        logger.info(
            f"Establishing Gemini SDK session with model: {self._model_name}")

        try:
            # Construct live_connect_kwargs for LiveConnectConfig by iterating through InputParams
            live_connect_kwargs = {}

            # Direct field mappings from InputParams to LiveConnectConfig
            direct_fields = [
                'generation_config', 'input_audio_transcription', 'max_output_tokens',
                'media_resolution', 'output_audio_transcription', 'realtime_input_config',
                'response_modalities', 'seed', 'session_resumption', 'temperature',
                'top_k', 'top_p'
            ]

            for field in direct_fields:
                value = getattr(self._input_params, field, None)
                if value is not None:
                    live_connect_kwargs[field] = value

            # Handle context_window_compression specially
            if (self._input_params.context_window_compression and
                    self._input_params.context_window_compression.enabled):

                cwc_kwargs = {}
                if self._input_params.context_window_compression.trigger_tokens is not None:
                    cwc_kwargs['trigger_tokens'] = self._input_params.context_window_compression.trigger_tokens

                # Handle sliding_window if present
                if self._input_params.context_window_compression.sliding_window:
                    sliding_window = SlidingWindow(
                        target_tokens=self._input_params.context_window_compression.sliding_window.target_tokens
                    )
                    cwc_kwargs['sliding_window'] = sliding_window

                live_connect_kwargs['context_window_compression'] = ContextWindowCompressionConfig(
                    **cwc_kwargs)

            # Handle speech_config - only create if audio output is requested
            if self._input_params.speech_config is not None:
                live_connect_kwargs['speech_config'] = self._input_params.speech_config
                logger.debug("Using provided speech_config")
            else:
                # Only create default speech_config if audio is in response_modalities
                response_modalities = getattr(
                    self._input_params, 'response_modalities', None)
                logger.debug(f"Response modalities: {response_modalities}")
                if response_modalities and any(mod.value == 'AUDIO' for mod in response_modalities):
                    voice_config = VoiceConfig(
                        prebuilt_voice_config=PrebuiltVoiceConfig(
                            voice_name=self._default_voice_id
                        )
                    )
                    live_connect_kwargs['speech_config'] = SpeechConfig(
                        voice_config=voice_config,
                        language_code=self._default_language_code
                    )
                    logger.debug(
                        f"Created default speech_config with voice: {self._default_voice_id}")
                else:
                    logger.warning(
                        "No audio modality detected - speech_config will not be created")

            # Handle system_instruction
            if self._system_instruction is not None:
                live_connect_kwargs['system_instruction'] = self._system_instruction

            # Handle tools
            if self._tools:
                tool_list_as_dicts = []
                if isinstance(self._tools, ToolsSchema):
                    if self._adapter_instance:
                        tool_list_as_dicts = self._adapter_instance.to_provider_tools_format(
                            self._tools)
                    else:
                        logger.warning(
                            "GeminiLLMAdapter instance not available for tool conversion during connect.")
                elif isinstance(self._tools, list):
                    tool_list_as_dicts = self._tools

                if tool_list_as_dicts:
                    try:
                        gemini_tools_list = [Tool(**td)
                                             for td in tool_list_as_dicts]
                        live_connect_kwargs['tools'] = gemini_tools_list
                    except Exception as e:
                        logger.error(
                            f"Error converting tool dictionaries to genai.types.Tool objects: {e}")

            # Filter out None values to avoid issues with SDK
            live_connect_kwargs = {
                k: v for k, v in live_connect_kwargs.items() if v is not None}

            self._session_closed_flag = False  # Reset for this loop execution

            live_config = LiveConnectConfig(**live_connect_kwargs)

            logger.debug(f"Connecting with LiveConnectConfig: {live_config}")

            self._session_connector_obj = self._client.aio.live.connect(
                model=self._model_name,
                config=live_config,
            )
            self._session = await self._session_connector_obj.__aenter__()
            logger.info("Gemini SDK session established successfully.")
            if not self._llm_task or self._llm_task.done():
                logger.info(
                    "Gemini SDK: Starting send/receive loop from _establish_sdk_session.")
                self._llm_task = self.create_task(
                    self._sdk_send_receive_loop())
            else:
                logger.info(
                    "Gemini SDK: Send/receive loop already running or pending from _establish_sdk_session call.")

        except Exception as e:
            logger.opt(exception=e).error(
                "Failed to establish Gemini SDK session.")
            await self.push_frame(ErrorFrame(f"Gemini SDK connection error: {e}"))
            self._session = None
            self._session_connector_obj = None
            raise

    async def _disconnect(self):
        logger.info("Disconnecting from Gemini service")
        try:
            await self.stop_all_metrics()
            if hasattr(self, "_llm_task") and self._llm_task and not self._llm_task.done():
                logger.debug(
                    "Cancelling main LLM task for disconnection prior to session exit.")
                await self.cancel_task(self._llm_task)
            self._llm_task = None

            if hasattr(self, "_session_connector_obj") and self._session_connector_obj:
                logger.debug("Exiting SDK session context.")
                await self._session_connector_obj.__aexit__(None, None, None)

        except Exception as e:
            logger.opt(exception=e).error(
                "Error during Gemini disconnect process.")
        finally:
            # Ensure these are cleared even if __aexit__ fails or was skipped
            self._session = None
            if hasattr(self, "_session_connector_obj"):
                self._session_connector_obj = None
            logger.info("Gemini service disconnected.")

    async def _sdk_send_receive_loop(self):
        self._session_closed_flag = False
        if not self._session:
            logger.error(
                "SDK session not established. Cannot start send/receive loop.")
            await self.push_frame(ErrorFrame("Gemini SDK session not ready."))
            return

        try:
            logger.info("Gemini SDK: _sdk_send_receive_loop started.")

            async def sender():
                logger.info("Gemini SDK: Sender task started.")
                while True:
                    try:
                        content_to_send = await self._outgoing_sdk_queue.get()
                        if content_to_send is None:  # Sentinel for stopping
                            self._outgoing_sdk_queue.task_done()
                            break
                        if self._session:
                            if isinstance(content_to_send, dict) and content_to_send.get("type") == "media_input":
                                media_payload = content_to_send.get("payload")
                                if media_payload:
                                    await self._session.send(input=media_payload)
                                else:
                                    logger.warning(
                                        "Gemini SDK: media_input message is missing payload.")
                            # Assuming text/turn based inputs are Content objects
                            elif isinstance(content_to_send, Content):
                                logger.info(
                                    f"Gemini SDK: Sender calling send_client_content with: {content_to_send}")
                                await self._session.send_client_content(turns=[content_to_send])
                            else:
                                logger.warning(
                                    f"Gemini SDK: Unknown content type in outgoing queue: {type(content_to_send)}. Not sending.")
                        else:
                            logger.warning(
                                "SDK sender: Session is None, cannot send.")
                            self._outgoing_sdk_queue.task_done()
                            break
                        self._outgoing_sdk_queue.task_done()
                    except asyncio.CancelledError:
                        logger.info("SDK sender task cancelled.")
                        break
                    except Exception as e:
                        logger.opt(exception=e).error(
                            "Error in SDK sender task.")
                        await self.push_frame(ErrorFrame(f"Gemini SDK send error: {e}"))
                        break
                logger.info("SDK sender task finished.")

            async def receiver():
                if not self._session:
                    logger.warning(
                        "Receiver: No active session to start receiving from.")
                    # Ensure sender is also stopped if receiver can't start
                    if hasattr(self, "_outgoing_sdk_queue") and self._outgoing_sdk_queue:
                        await self._outgoing_sdk_queue.put(None)
                    return

                logger.info("Gemini SDK: Receiver task started.")

                try:
                    while not self._session_closed_flag:
                        # Check if session was closed by another part (e.g. disconnect)
                        if not self._session:
                            logger.info(
                                "Receiver: Session became None, exiting receive loop.")
                            self._session_closed_flag = True  # Ensure outer loop terminates
                            break

                        logger.info(
                            "Gemini SDK: Receiver calling self._session.receive() for next turn/messages...")
                        try:
                            current_turn_iterator = self._session.receive()
                        except Exception as e_recv_init:
                            # Catch errors from the receive() call itself (e.g., if session is already dead)
                            logger.opt(exception=e_recv_init).error(
                                "Error calling self._session.receive(). Ending session.")
                            self._session_closed_flag = True
                            await self.push_frame(ErrorFrame(f"Gemini SDK error initiating receive: {e_recv_init}"))
                            break  # Exit while loop

                        turn_had_messages = False
                        try:
                            async for message in current_turn_iterator:  # Iterate messages in this specific turn
                                turn_had_messages = True

                                # 1. Check for session-ending go_away first
                                if hasattr(message, 'go_away') and message.go_away:
                                    reason_obj = getattr(
                                        message.go_away, 'reason', None)
                                    reason_phrase = "No reason provided"
                                    if reason_obj:
                                        reason_phrase = getattr(
                                            reason_obj, 'name', str(reason_obj))
                                    logger.info(
                                        f"Gemini SDK server sent go_away: {reason_phrase}. Ending session.")
                                    await self.push_frame(ErrorFrame(f"Gemini session ending (go_away): {reason_phrase}"))
                                    self._session_closed_flag = True
                                    # Break from inner message loop (current turn)
                                    break

                                # 2. Process server_content (model responses, transcriptions, errors in content)
                                if hasattr(message, 'server_content') and message.server_content:
                                    sc = message.server_content

                                    if sc.model_turn and sc.model_turn.parts:
                                        for part in sc.model_turn.parts:
                                            if part.text is not None:  # Explicitly check for None as empty string is valid
                                                if not self._bot_text_buffer:
                                                    await self.push_frame(LLMFullResponseStartFrame())
                                                self._bot_text_buffer += part.text
                                                await self.push_frame(LLMTextFrame(text=part.text))

                                            if part.inline_data and part.inline_data.data and part.inline_data.mime_type.startswith("audio/"):
                                                audio_data = part.inline_data.data
                                                if not self._bot_is_speaking:
                                                    self._bot_is_speaking = True
                                                    await self.push_frame(TTSStartedFrame())
                                                    if not self._bot_text_buffer:  # If audio starts before any text
                                                        await self.push_frame(LLMFullResponseStartFrame())
                                                self._bot_audio_buffer.extend(
                                                    audio_data)
                                                await self.push_frame(TTSAudioRawFrame(
                                                    audio=audio_data, sample_rate=self._output_audio_sample_rate, num_channels=1
                                                ))

                                    if sc.input_transcription and sc.input_transcription.text is not None:
                                        transcript = sc.input_transcription.text
                                        logger.debug(
                                            f"[Transcription:user] {transcript} (partial)")
                                        await self.push_frame(TranscriptionFrame(
                                            # TODO: Check SDK for finality
                                            text=transcript, user_id="", timestamp=time_now_iso8601(), is_final=False
                                        ))

                                    if sc.output_transcription and sc.output_transcription.text is not None:
                                        transcript_text = sc.output_transcription.text
                                        logger.debug(
                                            f"[Transcription:bot] {transcript_text}")
                                        # Only add if it's new text, not just a repeat of model_turn.parts text
                                        if not self._bot_text_buffer or self._bot_text_buffer[-len(transcript_text):] != transcript_text:
                                            if not self._bot_text_buffer:
                                                await self.push_frame(LLMFullResponseStartFrame())
                                            # This might be redundant if model_turn.parts already provided the text.
                                            # Consider if this should append to _bot_text_buffer or be a separate frame type.
                                            # For now, appending if it seems new.
                                            self._bot_text_buffer += transcript_text
                                            await self.push_frame(LLMTextFrame(text=transcript_text))

                                    if sc.turn_complete:
                                        logger.debug(
                                            "SDK received server_content.turn_complete.")
                                        if self._bot_is_speaking:
                                            self._bot_is_speaking = False
                                            await self.push_frame(TTSStoppedFrame())
                                        if self._bot_text_buffer:  # If there was any text accumulated
                                            await self.push_frame(LLMFullResponseEndFrame())
                                        self._bot_text_buffer = ""
                                        self._bot_audio_buffer = bytearray()
                                        # This server turn is complete. The outer loop will call receive() for the next turn.

                                    if hasattr(sc, 'error') and sc.error:
                                        err_msg = getattr(
                                            sc.error, 'message', str(sc.error))
                                        logger.error(
                                            f"Gemini SDK server_content error: {err_msg}")
                                        await self.push_frame(ErrorFrame(f"Gemini server_content error: {err_msg}"))
                                        self._session_closed_flag = True
                                        break  # Break from inner message loop

                                # 3. Process top-level usage_metadata
                                if hasattr(message, 'usage_metadata') and message.usage_metadata:
                                    await self._handle_usage_metadata(message.usage_metadata)

                                # TODO: Handle other top-level message fields like tool_call, tool_call_cancellation if needed.

                        except asyncio.CancelledError:  # Inner loop cancellation
                            logger.info(
                                "SDK receiver's current turn iteration cancelled.")
                            # Assume cancellation means stop all.
                            self._session_closed_flag = True
                            # Re-raise to be caught by outer handler if necessary or stop task.
                            raise
                        except Exception as e_inner:
                            logger.opt(exception=e_inner).error(
                                "Error iterating current turn messages.")
                            await self.push_frame(ErrorFrame(f"Gemini SDK error processing turn: {e_inner}"))
                            # Assume error in turn processing is fatal for session.
                            self._session_closed_flag = True
                            # This break will exit the outer while loop because the flag is set.
                            break

                        # After inner loop (processing all messages of a single turn):
                        if self._session_closed_flag:
                            logger.info(
                                "Receiver: Session closed flag set during turn processing, exiting outer loop.")
                            break  # Break from outer while loop (all turns)

                        if not turn_had_messages and not self._session_closed_flag:
                            # This means self._session.receive() returned an empty iterator but didn't signal session closure.
                            # This could be an idle period. The SDK might handle this gracefully.
                            # If this leads to busy-looping, a small asyncio.sleep(0.01) might be needed here.
                            logger.debug(
                                "Gemini SDK: Current turn iterator completed with no messages (idle or end of stream for now).")
                            # The outer loop will call self._session.receive() again.

                except asyncio.CancelledError:
                    logger.info("SDK receiver task (outer loop) cancelled.")
                    self._session_closed_flag = True
                except Exception as e_outer:
                    logger.opt(exception=e_outer).error(
                        "Critical error in SDK receiver task's outer loop.")
                    await self.push_frame(ErrorFrame(f"Gemini SDK receive critical error: {e_outer}"))
                    self._session_closed_flag = True
                finally:
                    logger.info("SDK receiver task finished.")
                    if hasattr(self, "_outgoing_sdk_queue") and self._outgoing_sdk_queue:
                        if self._outgoing_sdk_queue:  # Check if it's not None
                            # Signal sender to stop
                            await self._outgoing_sdk_queue.put(None)

            sender_task = self.create_task(sender())
            receiver_task = self.create_task(receiver())

            done, pending = await asyncio.wait(
                [sender_task, receiver_task],
                return_when=asyncio.FIRST_COMPLETED,
            )

            for task_to_cancel in pending:
                if not task_to_cancel.done():
                    task_to_cancel.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

        except asyncio.CancelledError:
            logger.info("SDK send/receive loop cancelled.")
        except Exception as e:
            logger.opt(exception=e).error("Error in SDK send/receive loop.")
            await self.push_frame(ErrorFrame(f"Gemini SDK loop error: {e}"))
        finally:
            logger.info("Gemini SDK send/receive loop finished.")
            if self._session:
                # Session should be closed by _disconnect cancelling this task,
                # if _establish_sdk_session uses `async with` or if session has __aexit__.
                # For now, just nullify.
                pass
            self._session = None
            if self._bot_is_speaking:
                await self.push_frame(TTSStoppedFrame())
                self._bot_is_speaking = False
            if self._bot_text_buffer:
                await self.push_frame(LLMFullResponseEndFrame())
                self._bot_text_buffer = ""
            self._bot_audio_buffer = bytearray()

    #

    def create_context_aggregator(
        self,
        context: OpenAILLMContext,
        *,
        user_params: LLMUserAggregatorParams = LLMUserAggregatorParams(),
        assistant_params: LLMAssistantAggregatorParams = LLMAssistantAggregatorParams(),
    ) -> GeminiMultimodalLiveContextAggregatorPair:
        """Create an instance of GeminiMultimodalLiveContextAggregatorPair from
        an OpenAILLMContext. Constructor keyword arguments for both the user and
        assistant aggregators can be provided.

        Args:
            context (OpenAILLMContext): The LLM context.
            user_params (LLMUserAggregatorParams, optional): User aggregator
                parameters.
            assistant_params (LLMAssistantAggregatorParams, optional): User
                aggregator parameters.

        Returns:
            GeminiMultimodalLiveContextAggregatorPair: A pair of context
            aggregators, one for the user and one for the assistant,
            encapsulated in an GeminiMultimodalLiveContextAggregatorPair.

        """
        context.set_llm_adapter(self.get_llm_adapter())

        GeminiMultimodalLiveContext.upgrade(context)
        user = GeminiMultimodalLiveUserContextAggregator(
            context, params=user_params)

        assistant_params.expect_stripped_words = False
        assistant = GeminiMultimodalLiveAssistantContextAggregator(
            context, params=assistant_params)
        return GeminiMultimodalLiveContextAggregatorPair(_user=user, _assistant=assistant)
