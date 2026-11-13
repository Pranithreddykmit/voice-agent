"""
Speech-to-Text (STT) Provider Subsystem for Multi-Agent Telugu Conversational AI.
Multilingual Whisper ASR with bilingual academic prompt conditioning.
Supports Telugu, English, and Telugu-English code-mixed speech.
"""

import time
import logging
from abc import ABC, abstractmethod
from typing import Tuple, Optional
import numpy as np

from app.config import settings
from app.speech.audio import is_silence

logger = logging.getLogger(__name__)

# Bilingual academic domain vocabulary for Whisper prompt conditioning.
# Kept short (~40 tokens) so it does not eat into the max_new_tokens output budget.
# This guides vocabulary selection WITHOUT forcing a single language.
_ACADEMIC_PROMPT = (
    "attendance, semester exams, hostel, library, fee, medical leave, "
    "హాజరు, పరీక్షలు, ఫీజు, దరఖాస్తు."
)


class STTTranscriptionError(Exception):
    """Raised when speech-to-text transcription fails."""
    pass


class BaseSTTProvider(ABC):
    """Abstract Base Class for Speech-to-Text Providers."""

    @abstractmethod
    def transcribe(
        self,
        audio_array: np.ndarray,
        sample_rate: int = 16000,
        language: Optional[str] = None
    ) -> Tuple[str, str, float]:
        """
        Transcribes 1D mono audio array to text.

        Returns:
            (transcript: str, language: str, latency_ms: float)
        """
        pass


class MockSTTProvider(BaseSTTProvider):
    """
    Deterministic Mock STT Provider for unit tests, offline environments, and synthetic benchmarks.
    """

    def __init__(self, fixed_transcript: Optional[str] = None):
        self.fixed_transcript = fixed_transcript
        logger.info("Initialized MockSTTProvider.")

    def transcribe(
        self,
        audio_array: np.ndarray,
        sample_rate: int = 16000,
        language: Optional[str] = None
    ) -> Tuple[str, str, float]:
        t0 = time.perf_counter()

        if is_silence(audio_array):
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return "", language or "te", round(elapsed_ms, 2)

        if self.fixed_transcript:
            transcript = self.fixed_transcript
        else:
            duration = len(audio_array) / float(sample_rate)
            if duration < 1.5:
                transcript = "నమస్కారం"
                lang = "te"
            elif duration < 3.0:
                transcript = "What is 15% of 240?"
                lang = "en"
            else:
                transcript = "కనీస హాజరు ఎంత శాతం ఉండాలి?"
                lang = "te"

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return transcript, language or "te", round(elapsed_ms, 2)


class WhisperSTTProvider(BaseSTTProvider):
    """
    Multilingual Whisper ASR Provider using direct AutoModelForSpeechSeq2Seq generate.
    Uses bilingual academic prompt conditioning for Telugu + English code-mixed speech.
    Language detection is left to Whisper's native multilingual capacity.
    """

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        self.model_name = model_name or settings.STT_MODEL_NAME
        self.device = device or settings.STT_DEVICE
        self._processor = None
        self._model = None
        self._torch_device = "cpu"
        self._prompt_ids = None  # Bilingual academic prompt token ids
        self._load_model()

    def _load_model(self):
        t0 = time.perf_counter()
        logger.info(f"Loading Whisper ASR model '{self.model_name}' on device '{self.device}'...")
        try:
            import os
            import torch
            from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

            # Configure optimal multi-threading for Apple Silicon / multi-core CPU
            cpu_cores = os.cpu_count() or 4
            thread_count = min(8, cpu_cores)
            torch.set_num_threads(thread_count)

            req_dev = (self.device or "auto").lower()
            if (req_dev == "cuda" or req_dev == "auto") and torch.cuda.is_available():
                self._torch_device = "cuda"
                dtype = torch.float16
            elif (req_dev == "mps" or req_dev == "auto") and torch.backends.mps.is_available():
                self._torch_device = "mps"
                dtype = torch.float32
            else:
                self._torch_device = "cpu"
                dtype = torch.float32

            self._processor = AutoProcessor.from_pretrained(self.model_name)
            self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_name,
                dtype=dtype
            )
            self._model.to(self._torch_device)
            self._model.eval()

            # Configure generation parameters.
            # NOTE: repetition_penalty and no_repeat_ngram_size are intentionally removed.
            # They penalise natural Telugu agglutinative subwords and cause Devanagari
            # script bleed. Hallucination loops are handled by sanitize_transcript().
            # max_new_tokens=256: the prompt uses ~45 tokens leaving 210 tokens for output,
            # which comfortably covers all utterances in the benchmark (longest ~190 tokens).
            if hasattr(self._model, "generation_config"):
                self._model.generation_config.max_new_tokens = 256

            # Build and cache bilingual academic prompt ids
            self._prompt_ids = self._build_prompt_ids()

            param_count = sum(p.numel() for p in self._model.parameters())
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            logger.info(
                f"Whisper ASR Model Verification: status=LOADED | model='{self.model_name}' | "
                f"params={param_count/1e6:.1f}M | device='{self._torch_device}' | "
                f"threads={thread_count} | load_time={elapsed_ms:.2f}ms"
            )
        except Exception as e:
            logger.warning(
                f"Failed to load Whisper ASR model '{self.model_name}': {e}. "
                "Falling back to MockSTTProvider."
            )
            self._processor = None
            self._model = None

    def _build_prompt_ids(self):
        """
        Encodes the bilingual academic domain prompt into Whisper decoder token ids.
        The prompt guides vocabulary selection for Telugu + English code-mixed academic speech
        WITHOUT locking the model to a single language.
        Returns a torch.Tensor or None on failure.
        """
        import torch
        try:
            raw = self._processor.get_prompt_ids(_ACADEMIC_PROMPT)
            if isinstance(raw, np.ndarray):
                return torch.from_numpy(raw).to(self._torch_device)
            return torch.tensor(raw, device=self._torch_device)
        except Exception as e:
            logger.warning(f"[Whisper STT] Could not build prompt ids: {e}. Proceeding without prompt.")
            return None

    def transcribe(
        self,
        audio_array: np.ndarray,
        sample_rate: int = 16000,
        language: Optional[str] = None
    ) -> Tuple[str, str, float]:
        t0 = time.perf_counter()
        duration_sec = len(audio_array) / float(sample_rate) if sample_rate > 0 else 0.0

        if is_silence(audio_array):
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            logger.info(f"[Whisper STT] Detected silence/empty audio ({duration_sec:.2f}s). Skipping transcription.")
            return "", language or "te", round(elapsed_ms, 2)

        if self._model is None or self._processor is None:
            mock = MockSTTProvider()
            return mock.transcribe(audio_array, sample_rate, language)

        # Ensure mono 1D float32 array
        if audio_array.ndim > 1:
            audio_array = np.mean(audio_array, axis=1)

        # Automatic polyphase resampling to 16kHz if needed
        if sample_rate != 16000 and sample_rate > 0:
            import math, scipy.signal
            gcd = math.gcd(int(sample_rate), 16000)
            up = 16000 // gcd
            down = int(sample_rate) // gcd
            audio_array = scipy.signal.resample_poly(audio_array, up, down).astype(np.float32)
            sample_rate = 16000

        try:
            import torch

            # 1. Feature extraction
            inputs = self._processor(
                audio_array,
                sampling_rate=16000,
                return_tensors="pt"
            )
            input_features = inputs.input_features.to(self._torch_device)

            # 2. Language handling.
            # If the caller or config specifies an explicit language (not "auto"), honour it.
            # Otherwise do NOT force a language — let Whisper's native multilingual decoder
            # handle Telugu, English, and intra-sentence code-switching naturally.
            explicit_lang = language or settings.STT_LANGUAGE
            force_language = (
                explicit_lang.lower()
                if explicit_lang and explicit_lang.lower() not in ["auto", "none", ""]
                else None
            )

            # 3. Generate with bilingual academic prompt conditioning.
            # prompt_ids conditions Whisper's vocabulary priors without forcing a language label.
            # task="transcribe" ensures faithful output (not translation).
            # No repetition_penalty or no_repeat_ngram_size — these were found to destroy
            # natural Telugu agglutinative subwords.
            generate_kwargs = dict(
                task="transcribe",
                prompt_ids=self._prompt_ids,  # bilingual academic vocab conditioning
            )
            if force_language:
                generate_kwargs["language"] = force_language

            with torch.no_grad():
                out = self._model.generate(input_features, **generate_kwargs)

            # 4. Decode text cleanly
            raw_transcript = self._processor.batch_decode(
                out,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0].strip()

            transcript = self.sanitize_transcript(raw_transcript)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            # Determine what language Whisper actually used (for logging/return value)
            detected_lang = force_language or "multilingual"

            logger.info(
                f"[Whisper STT Request] duration={duration_sec:.2f}s | sr={sample_rate}Hz | "
                f"lang='{detected_lang}' | raw='{raw_transcript[:80]}' | "
                f"transcript='{transcript[:80]}' | latency={elapsed_ms:.2f}ms"
            )

            return transcript, detected_lang, round(elapsed_ms, 2)

        except Exception as e:
            logger.error(f"Whisper transcription failed: {e}", exc_info=True)
            raise STTTranscriptionError(f"Speech transcription error: {str(e)}")

    @staticmethod
    def sanitize_transcript(text: str) -> str:
        """Collapse and filter repetition loops and hallucination spirals."""
        import re
        if not text:
            return ""
        # 1. Collapse repeating characters/syllables (e.g. కాకాకాకా... -> కా or "")
        collapsed = re.sub(r"(.{1,6}?)\1{3,}", r"\1", text).strip()
        # 2. Collapse repeating word phrases (e.g. hello hello hello world -> hello world)
        collapsed = re.sub(r"\b(\w+)(?:\s+\1\b){2,}", r"\1", collapsed, flags=re.IGNORECASE).strip()
        # 3. If remaining string collapsed to 2 or fewer characters, it is an acoustic hallucination
        if len(collapsed) <= 2:
            return ""
        return collapsed


_stt_instance: Optional[BaseSTTProvider] = None


def get_stt_provider(engine: Optional[str] = None) -> BaseSTTProvider:
    """Factory function for STT provider singleton."""
    global _stt_instance
    active_engine = (engine or settings.STT_ENGINE).lower()

    if _stt_instance is not None:
        return _stt_instance

    if active_engine == "mock":
        _stt_instance = MockSTTProvider()
    elif active_engine in ["whisper_turbo", "whisper", "indicconformer"]:
        _stt_instance = WhisperSTTProvider(model_name=settings.STT_MODEL_NAME)
    else:
        logger.warning(f"Unknown STT engine '{active_engine}'. Defaulting to MockSTTProvider.")
        _stt_instance = MockSTTProvider()

    return _stt_instance
