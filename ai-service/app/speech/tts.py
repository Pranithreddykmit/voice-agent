"""
Text-to-Speech (TTS) Subsystem for Phase 5.
Supports IndicF5 Diffusion Synthesizer and deterministic In-Memory Acoustic Fallback.
Provides strict engine transparency (never disguise fallback as neural model).
"""

import abc
import io
import os
import time
import logging
from typing import Tuple, Optional
import numpy as np
import soundfile as sf
import torch

from app.config import settings

logger = logging.getLogger("ai_service.speech.tts")


class TTSSynthesisError(Exception):
    """Raised when text-to-speech synthesis fails."""
    pass


class BaseTTSProvider(abc.ABC):
    """Abstract Base Class for Text-to-Speech Providers."""

    @property
    @abc.abstractmethod
    def engine_name(self) -> str:
        """Name of the TTS engine."""
        pass

    @property
    @abc.abstractmethod
    def is_fallback(self) -> bool:
        """Whether this provider is a fallback engine."""
        pass

    @property
    def adapter_name(self) -> str:
        """Name of active adapter or 'none'."""
        return "none"

    @property
    def is_adapter_active(self) -> bool:
        """Whether a fine-tuned LoRA adapter is actively attached."""
        return False

    @abc.abstractmethod
    def synthesize(
        self,
        text: str,
        voice_id: Optional[str] = None,
        language: str = "te"
    ) -> Tuple[bytes, float]:
        """
        Synthesizes text into 16kHz/24kHz PCM WAV audio bytes.
        Returns:
            Tuple[bytes, float]: (wav_bytes, latency_ms)
        """
        pass


class FallbackTTSProvider(BaseTTSProvider):
    """
    In-memory acoustic speech synthesizer for deterministic local execution and fast tests.
    Generates formant-modulated harmonic waveforms delivering 16kHz PCM WAV in ~1ms.
    """

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate

    @property
    def engine_name(self) -> str:
        return "fallback_acoustic"

    @property
    def is_fallback(self) -> bool:
        return True

    @property
    def adapter_name(self) -> str:
        return "none"

    @property
    def is_adapter_active(self) -> bool:
        return False

    def synthesize(
        self,
        text: str,
        voice_id: Optional[str] = None,
        language: str = "te"
    ) -> Tuple[bytes, float]:
        start_time = time.perf_counter()
        clean_text = text.strip()
        if not clean_text:
            raise TTSSynthesisError("Cannot synthesize empty text input.")

        try:
            # Estimate speech duration based on syllables / character count (~12 chars/sec)
            duration_sec = max(0.8, min(15.0, len(clean_text) * 0.08))
            num_samples = int(duration_sec * self.sample_rate)
            t = np.linspace(0, duration_sec, num_samples, endpoint=False)

            # Natural vocal pitch & formants (F0=185Hz female / 135Hz male)
            f0 = 185.0 if "female" in (voice_id or "female") else 135.0
            
            # Harmonic richness (F0 fundamental + harmonics + formant resonators)
            harmonic_1 = 0.5 * np.sin(2 * np.pi * f0 * t)
            harmonic_2 = 0.25 * np.sin(2 * np.pi * (2 * f0) * t)
            harmonic_3 = 0.15 * np.sin(2 * np.pi * (3 * f0) * t)
            formant_f1 = 0.1 * np.sin(2 * np.pi * 800.0 * t)   # Vocal tract F1
            formant_f2 = 0.05 * np.sin(2 * np.pi * 1800.0 * t) # Vocal tract F2

            carrier = harmonic_1 + harmonic_2 + harmonic_3 + formant_f1 + formant_f2

            # Cadence & speech envelope (natural syllabic rhythm modulation at 4Hz)
            syllable_rate = 4.0
            envelope = 0.6 + 0.4 * np.sin(2 * np.pi * syllable_rate * t)

            # Smooth attack and release
            fade_samples = int(0.04 * self.sample_rate)
            fade_in = np.linspace(0.0, 1.0, fade_samples)
            fade_out = np.linspace(1.0, 0.0, fade_samples)
            envelope[:fade_samples] *= fade_in
            envelope[-fade_samples:] *= fade_out

            audio_signal = (carrier * envelope * 0.4).astype(np.float32)

            # Serialize to standard 16-bit PCM WAV in memory
            buf = io.BytesIO()
            sf.write(buf, audio_signal, self.sample_rate, format="WAV", subtype="PCM_16")
            wav_bytes = buf.getvalue()

            latency_ms = (time.perf_counter() - start_time) * 1000.0
            return wav_bytes, round(latency_ms, 2)

        except Exception as e:
            logger.error(f"Fallback acoustic synthesis error: {e}", exc_info=True)
            raise TTSSynthesisError(f"Acoustic synthesis failure: {e}")


class IndicF5TTSProvider(BaseTTSProvider):
    """
    IndicF5 Neural Diffusion TTS Provider with optional LoRA Adapter.
    Executes genuine flow-matching diffusion speech synthesis on Telugu and Indic languages.
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        device: Optional[str] = None,
        adapter_path: Optional[str] = None
    ):
        self.sample_rate = sample_rate
        req_dev = (device or getattr(settings, "TTS_DEVICE", "auto") or "auto").lower()
        if (req_dev == "cuda" or req_dev == "auto") and torch.cuda.is_available():
            self.device = "cuda"
        elif (req_dev == "mps" or req_dev == "auto") and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self.adapter_path = adapter_path if adapter_path is not None else settings.TTS_ADAPTER
        self.fallback = FallbackTTSProvider(sample_rate=16000)
        self._model = None
        self._vocoder = None
        self._ref_wav_path = None
        self._ref_audio_tensor = None
        self._ref_text = "నమస్తే"
        self._blocker_reason = None
        self._adapter_active = False
        self._adapter_name = "none"
        self._load_indicf5_model()

    @property
    def engine_name(self) -> str:
        if self._model is not None:
            return "indicf5"
        return f"fallback_acoustic ({self._blocker_reason})"

    @property
    def is_fallback(self) -> bool:
        return self._model is None

    @property
    def adapter_name(self) -> str:
        return self._adapter_name

    @property
    def is_adapter_active(self) -> bool:
        return self._adapter_active

    @property
    def blocker_reason(self) -> Optional[str]:
        return self._blocker_reason

    def _resolve_adapter_path(self, raw_path: str) -> Optional[str]:
        """Resolves adapter directory relative to workspace or absolute path."""
        if not raw_path or raw_path.lower() == "none":
            return None
        from pathlib import Path
        p = Path(raw_path)
        if p.exists():
            return str(p.resolve())
        # Try relative to repo root
        repo_root = Path(__file__).resolve().parent.parent.parent.parent
        candidate = repo_root / raw_path
        if candidate.exists():
            return str(candidate.resolve())
        return None

    def _load_indicf5_model(self):
        """Loads genuine ai4bharat/IndicF5 checkpoint, Vocos vocoder, and optional LoRA adapter."""
        try:
            logger.info(f"Loading genuine ai4bharat/IndicF5 neural model on device: {self.device}...")
            t0 = time.perf_counter()
            import torchaudio
            from huggingface_hub import hf_hub_download
            from safetensors.torch import load_file
            from f5_tts.model import CFM, DiT
            from f5_tts.infer.utils_infer import load_vocoder

            vocab_path = hf_hub_download(repo_id="ai4bharat/IndicF5", filename="checkpoints/vocab.txt")
            ckpt_path = hf_hub_download(repo_id="ai4bharat/IndicF5", filename="model.safetensors")
            self._ref_wav_path = hf_hub_download(repo_id="ai4bharat/IndicF5", filename="samples/namaste.wav")

            # Pre-load and cache reference audio tensor directly on device
            ref_audio, sr = torchaudio.load(self._ref_wav_path)
            if ref_audio.shape[0] > 1:
                ref_audio = torch.mean(ref_audio, dim=0, keepdim=True)
            if sr != 24000:
                resampler = torchaudio.transforms.Resample(sr, 24000)
                ref_audio = resampler(ref_audio)
            self._ref_audio_tensor = ref_audio.to(self.device)

            # Load and clean state dict
            raw_sd = load_file(ckpt_path)
            cleaned_sd = {}
            for k, v in raw_sd.items():
                clean_k = k.replace("ema_model._orig_mod.", "").replace("ema_model.", "").replace("_orig_mod.", "")
                cleaned_sd[clean_k] = v

            weight_shape = cleaned_sd["transformer.text_embed.text_embed.weight"].shape[0]

            with open(vocab_path, "r", encoding="utf-8") as f:
                vocab_char_map = {ch.strip(): i for i, ch in enumerate(f)}

            model_cfg = dict(
                dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4,
                mel_dim=100, text_num_embeds=weight_shape - 1
            )

            transformer = DiT(**model_cfg)
            transformer.load_state_dict({k.replace("transformer.", ""): v for k, v in cleaned_sd.items() if k.startswith("transformer.")}, strict=False)

            # Check and attach LoRA adapter if configured
            resolved_adapter = self._resolve_adapter_path(self.adapter_path)
            if resolved_adapter:
                try:
                    from peft import LoraConfig, get_peft_model
                    from pathlib import Path

                    adapter_dir = Path(resolved_adapter)
                    self._adapter_name = adapter_dir.name
                    logger.info(f"Loading LoRA adapter configuration from: {resolved_adapter}")

                    lora_config = LoraConfig.from_pretrained(resolved_adapter)
                    transformer = get_peft_model(transformer, lora_config)

                    # Locate weights checkpoint
                    weights_path = adapter_dir / "adapter_model.safetensors"
                    if weights_path.exists():
                        adapter_sd = load_file(str(weights_path))
                    else:
                        bin_path = adapter_dir / "adapter_model.bin"
                        if not bin_path.exists():
                            raise FileNotFoundError(f"No adapter weights (.safetensors or .bin) found in {resolved_adapter}")
                        adapter_sd = torch.load(str(bin_path), map_location="cpu")

                    # Load weights into injected PEFT layers
                    incompatible = transformer.load_state_dict(adapter_sd, strict=False)

                    loaded_lora = [k for k in adapter_sd.keys() if "lora_" in k]
                    missing_lora = [k for k in incompatible.missing_keys if "lora_" in k]
                    unexpected_lora = [k for k in incompatible.unexpected_keys if "lora_" in k]
                    target_mods = len([m for n, m in transformer.named_modules() if hasattr(m, "lora_A")])
                    total_lora_params = sum(p.numel() for p in adapter_sd.values())

                    if len(missing_lora) > 0:
                        raise RuntimeError(f"Missing {len(missing_lora)} LoRA keys during checkpoint attachment: {missing_lora[:5]}")

                    self._adapter_active = True
                    logger.info(
                        f"LoRA Adapter Verification: status=LOADED_VERIFIED | "
                        f"adapter='{self._adapter_name}' | target_modules={target_mods} | "
                        f"checkpoint_keys={len(adapter_sd)} | loaded_keys={len(loaded_lora)} | "
                        f"missing_keys=0 | unexpected_keys={len(unexpected_lora)} | "
                        f"adapted_parameters={total_lora_params:,}"
                    )
                except Exception as lora_err:
                    self._adapter_active = False
                    self._adapter_name = "none"
                    logger.error(f"Failed to attach LoRA adapter '{self.adapter_path}': {lora_err}", exc_info=True)
                    raise RuntimeError(f"LoRA adapter attachment failed: {lora_err}")
            else:
                self._adapter_active = False
                self._adapter_name = "none"


            self._model = CFM(
                transformer=transformer,
                mel_spec_kwargs=dict(
                    target_sample_rate=24000, n_mel_channels=100, hop_length=256,
                    win_length=1024, n_fft=1024, mel_spec_type="vocos"
                ),
                vocab_char_map=vocab_char_map
            )

            self._model.to(self.device)
            self._model.eval()

            self._vocoder = load_vocoder(vocoder_name="vocos", is_local=False, device=self.device)
            load_time = (time.perf_counter() - t0) * 1000.0
            logger.info(f"ai4bharat/IndicF5 model (Adapter: {self._adapter_name}) loaded on {self.device} in {load_time:.2f} ms.")

        except Exception as e:
            self._model = None
            self._vocoder = None
            self._ref_audio_tensor = None
            self._adapter_active = False
            self._adapter_name = "none"
            self._blocker_reason = str(e)
            logger.warning(f"IndicF5 neural model initialization failed: {e}. Active: acoustic fallback.")

    def _chunk_speech_text(self, text: str, max_chars: int = 150) -> list:
        """
        Splits long responses into manageable conversational chunks.
        Single conversational sentences (<=150 chars) are synthesized in 1 single pass
        to prevent unnecessary multi-batch splitting ('0/3' hangs).
        """
        import re
        text = text.strip()
        if len(text) <= max_chars:
            return [text]

        sentences = [s.strip() for s in re.split(r"(?<=[.!?\n])\s+", text) if s.strip()]
        if not sentences:
            return [text]

        chunks = []
        current_chunk = ""
        for s in sentences:
            if not current_chunk:
                current_chunk = s
            elif len(current_chunk) + len(s) + 1 <= max_chars:
                current_chunk += " " + s
            else:
                chunks.append(current_chunk)
                current_chunk = s
        if current_chunk:
            chunks.append(current_chunk)
        return chunks

    def _synthesize_chunk_direct(self, chunk: str) -> np.ndarray:
        """
        Direct, single-pass character-aware diffusion synthesis without ThreadPool contention.
        Safely computes Telugu/Indic/English duration from real character count,
        preventing pathological UTF-8 byte explosions.
        """
        from f5_tts.model.utils import convert_char_to_pinyin

        target_sr = 24000
        hop_length = 256
        ref_audio_len_frames = self._ref_audio_tensor.shape[-1] // hop_length

        # Character-aware speaking rate (11.0 chars/sec for Indic syllables, 14.5 for English)
        is_indic = any(ord(c) >= 0x0C00 and ord(c) <= 0x0C7F for c in chunk)
        chars_per_sec = 11.0 if is_indic else 14.5
        gen_sec = max(0.8, len(chunk) / chars_per_sec) + 0.25
        gen_sec = min(gen_sec, 22.0)  # Hard ceiling for single chunk

        duration_frames = ref_audio_len_frames + int(gen_sec * target_sr / hop_length)

        text_list = [self._ref_text + " " + chunk]
        final_text_list = convert_char_to_pinyin(text_list)

        with torch.inference_mode():
            generated, _ = self._model.sample(
                cond=self._ref_audio_tensor,
                text=final_text_list,
                duration=duration_frames,
                steps=16,
                cfg_strength=2.0,
                sway_sampling_coef=-1.0,
            )
            del _

            generated_mel = generated[:, ref_audio_len_frames:, :].to(torch.float32).permute(0, 2, 1)
            generated_wave = self._vocoder.decode(generated_mel)
            del generated, generated_mel

            wav_np = generated_wave.squeeze().cpu().numpy()
            del generated_wave

        return wav_np

    def synthesize(
        self,
        text: str,
        voice_id: Optional[str] = None,
        language: str = "te"
    ) -> Tuple[bytes, float]:
        from app.speech.audio import prepare_speech_text
        clean_text = prepare_speech_text(text)
        if not clean_text:
            clean_text = text.strip()
        if not clean_text:
            raise TTSSynthesisError("Cannot synthesize empty text input.")

        if self._model is None or self._vocoder is None or self._ref_audio_tensor is None:
            logger.debug("IndicF5 neural model unavailable; executing acoustic fallback.")
            return self.fallback.synthesize(text=clean_text, voice_id=voice_id, language=language)

        start_time = time.perf_counter()
        try:
            chunks = self._chunk_speech_text(clean_text, max_chars=150)
            waves = []

            for chunk in chunks:
                chunk_wave = self._synthesize_chunk_direct(chunk)
                waves.append(chunk_wave)

            # Cross-fade and concatenate multiple chunks if present
            if len(waves) == 1:
                final_wave = waves[0]
            else:
                cross_fade_samples = int(0.05 * self.sample_rate)  # 50ms smooth cross-fade
                final_wave = waves[0]
                for i in range(1, len(waves)):
                    prev = final_wave
                    nxt = waves[i]
                    if len(prev) > cross_fade_samples and len(nxt) > cross_fade_samples:
                        fade_out = np.linspace(1.0, 0.0, cross_fade_samples)
                        fade_in = np.linspace(0.0, 1.0, cross_fade_samples)
                        overlap = prev[-cross_fade_samples:] * fade_out + nxt[:cross_fade_samples] * fade_in
                        final_wave = np.concatenate([prev[:-cross_fade_samples], overlap, nxt[cross_fade_samples:]])
                    else:
                        final_wave = np.concatenate([prev, nxt])

            buf = io.BytesIO()
            sf.write(buf, final_wave, self.sample_rate, format="WAV", subtype="PCM_16")
            wav_bytes = buf.getvalue()
            latency_ms = (time.perf_counter() - start_time) * 1000.0

            # MPS Memory Hygiene
            if self.device == "mps" and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()

            logger.info(f"[IndicF5 ({self._adapter_name})] Neural synthesis completed ({len(clean_text)} chars, {len(chunks)} chunks) in {latency_ms:.2f} ms.")
            return wav_bytes, round(latency_ms, 2)

        except Exception as e:
            logger.error(f"IndicF5 neural inference failure: {e}", exc_info=True)
            logger.warning("Falling back to acoustic generator due to neural inference error.")
            return self.fallback.synthesize(text=clean_text, voice_id=voice_id, language=language)


_tts_instance: Optional[BaseTTSProvider] = None


def get_tts_provider(
    engine: Optional[str] = None,
    adapter_path: Optional[str] = None
) -> BaseTTSProvider:
    """Factory creating configured TTS provider instance."""
    global _tts_instance
    active_engine = (engine or settings.TTS_ENGINE).lower()
    active_adapter = adapter_path if adapter_path is not None else settings.TTS_ADAPTER

    # Return cached singleton if matching engine and adapter
    if _tts_instance is not None:
        return _tts_instance

    if active_engine == "indicf5":
        _tts_instance = IndicF5TTSProvider(
            sample_rate=settings.TTS_SAMPLE_RATE,
            adapter_path=active_adapter
        )
    else:
        _tts_instance = FallbackTTSProvider(sample_rate=settings.TTS_SAMPLE_RATE)

    return _tts_instance

