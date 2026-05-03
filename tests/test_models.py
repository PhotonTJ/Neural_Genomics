"""
Unit tests for ndna.models module.

Uses GPT-2 (small, no auth needed) for integration tests.
"""

import pytest
import torch

from ndna.models import (
    ModelHandler,
    MODEL_REGISTRY,
    ARCH_DETECTION,
    ARCH_TO_REGISTRY,
    detect_architecture,
    get_registry_config,
    logit_lens,
    logit_lens_probs,
    batch_logit_lens,
    layerwise_predictions,
)


# -----------------------------------------------------------------------------
# Registry Tests
# -----------------------------------------------------------------------------

class TestRegistry:
    """Tests for the model registry."""

    def test_model_registry_structure(self):
        """Test that MODEL_REGISTRY has expected structure."""
        assert isinstance(MODEL_REGISTRY, dict)
        assert len(MODEL_REGISTRY) > 0

        # Check required keys in each entry
        required_keys = {"model_attr", "layers_attr", "norm_attr", "lm_head_attr", "embed_attr"}
        for arch, config in MODEL_REGISTRY.items():
            assert isinstance(config, dict), f"Config for {arch} is not a dict"
            missing = required_keys - set(config.keys())
            assert len(missing) == 0 or "pos_embed_attr" in config.keys(), \
                f"Missing keys {missing} in {arch} config"

    def test_arch_detection_structure(self):
        """Test that ARCH_DETECTION has expected structure."""
        assert isinstance(ARCH_DETECTION, dict)
        assert len(ARCH_DETECTION) > 0

        for arch, keywords in ARCH_DETECTION.items():
            assert isinstance(keywords, list), f"Keywords for {arch} is not a list"
            assert len(keywords) > 0, f"No keywords for {arch}"

    def test_arch_to_registry_mapping(self):
        """Test that all detection archs map to registry keys."""
        for arch in ARCH_DETECTION.keys():
            registry_key = ARCH_TO_REGISTRY.get(arch, arch)
            assert registry_key in MODEL_REGISTRY, \
                f"Architecture {arch} maps to {registry_key} which is not in MODEL_REGISTRY"


class TestArchitectureDetection:
    """Tests for architecture detection."""

    @pytest.mark.parametrize("model_type,model_name,expected", [
        ("gpt2", None, "gpt2"),
        (None, "gpt2-medium", "gpt2"),
        ("llama", None, "llama"),
        (None, "meta-llama/Llama-2-7b-hf", "llama"),
        ("mistral", None, "mistral"),
        (None, "mistralai/Mistral-7B-v0.1", "mistral"),
        ("phi", None, "phi"),
        (None, "microsoft/phi-2", "phi"),
        ("qwen2", "Qwen/Qwen2-7B", "qwen"),
        ("gemma", None, "gemma"),
        (None, "deepseek-ai/deepseek-llm-7b", "deepseek"),
        ("falcon", None, "falcon"),
        ("gpt_neox", None, "gpt_neox"),
        (None, "EleutherAI/pythia-410m", "gpt_neox"),
    ])
    def test_detect_architecture(self, model_type, model_name, expected):
        """Test architecture detection from model_type and model_name."""
        result = detect_architecture(model_type, model_name)
        assert result == expected

    def test_detect_architecture_unknown(self):
        """Test that unknown architecture raises ValueError."""
        with pytest.raises(ValueError, match="Could not detect architecture"):
            detect_architecture("unknown_model_type", "unknown/model-name")

    def test_get_registry_config(self):
        """Test getting registry config for known architectures."""
        config = get_registry_config("llama")
        assert config["model_attr"] == "model"
        assert config["layers_attr"] == "layers"
        assert config["norm_attr"] == "norm"

        config = get_registry_config("gpt2")
        assert config["model_attr"] == "transformer"
        assert config["layers_attr"] == "h"
        assert config["norm_attr"] == "ln_f"

    def test_get_registry_config_with_mapping(self):
        """Test getting config for architectures that map to others."""
        # deepseek maps to llama
        config = get_registry_config("deepseek")
        assert config["model_attr"] == "model"
        assert config["layers_attr"] == "layers"


# -----------------------------------------------------------------------------
# ModelHandler Tests (requires model loading)
# -----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def gpt2_handler():
    """Load GPT-2 model handler (shared across tests)."""
    return ModelHandler(
        "gpt2",
        device="cpu",
        dtype=torch.float32,
    )


class TestModelHandler:
    """Tests for ModelHandler with actual GPT-2 model."""

    def test_handler_initialization(self, gpt2_handler):
        """Test that handler initializes correctly."""
        assert gpt2_handler.model is not None
        assert gpt2_handler.tokenizer is not None
        assert gpt2_handler.model_name == "gpt2"

    def test_architecture_property(self, gpt2_handler):
        """Test architecture detection property."""
        assert gpt2_handler.architecture == "gpt2"

    def test_is_causal_property(self, gpt2_handler):
        """Test is_causal property."""
        assert gpt2_handler.is_causal is True

    def test_num_layers_property(self, gpt2_handler):
        """Test num_layers property."""
        # GPT-2 small has 12 layers
        assert gpt2_handler.num_layers == 12

    def test_vocab_size_property(self, gpt2_handler):
        """Test vocab_size property."""
        # GPT-2 has 50257 tokens
        assert gpt2_handler.vocab_size == 50257

    def test_hidden_size_property(self, gpt2_handler):
        """Test hidden_size property."""
        # GPT-2 small has hidden size 768
        assert gpt2_handler.hidden_size == 768

    def test_device_property(self, gpt2_handler):
        """Test device property."""
        assert gpt2_handler.device == "cpu"

    def test_get_layers(self, gpt2_handler):
        """Test get_layers returns correct number of layers."""
        layers = gpt2_handler.get_layers()
        assert len(layers) == 12

    def test_get_final_norm(self, gpt2_handler):
        """Test get_final_norm returns a module."""
        norm = gpt2_handler.get_final_norm()
        assert isinstance(norm, torch.nn.Module)

    def test_get_lm_head(self, gpt2_handler):
        """Test get_lm_head returns a module."""
        head = gpt2_handler.get_lm_head()
        assert isinstance(head, torch.nn.Module)

    def test_get_embeddings(self, gpt2_handler):
        """Test get_embeddings returns a module."""
        emb = gpt2_handler.get_embeddings()
        assert isinstance(emb, torch.nn.Module)

    def test_repr(self, gpt2_handler):
        """Test string representation."""
        repr_str = repr(gpt2_handler)
        assert "ModelHandler" in repr_str
        assert "gpt2" in repr_str
        assert "12" in repr_str  # num_layers


class TestModelHandlerOperations:
    """Tests for ModelHandler core operations."""

    def test_get_hidden_states(self, gpt2_handler):
        """Test get_hidden_states returns correct structure."""
        text = "Hello, world!"
        tokens = gpt2_handler.tokenizer(text, return_tensors="pt")
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]

        hidden_states = gpt2_handler.get_hidden_states(input_ids, attention_mask)

        # Should have num_layers + 1 hidden states (includes embedding)
        assert len(hidden_states) == 13  # 12 layers + embedding

        # Check shapes
        batch_size, seq_len = input_ids.shape
        hidden_size = gpt2_handler.hidden_size
        for h in hidden_states:
            assert h.shape == (batch_size, seq_len, hidden_size)

    def test_get_hidden_states_no_mask(self, gpt2_handler):
        """Test get_hidden_states works without attention mask."""
        text = "Test input"
        tokens = gpt2_handler.tokenizer(text, return_tensors="pt")
        input_ids = tokens["input_ids"]

        hidden_states = gpt2_handler.get_hidden_states(input_ids)
        assert len(hidden_states) == 13

    def test_logit_lens(self, gpt2_handler):
        """Test logit_lens produces correct output shape."""
        hidden_size = gpt2_handler.hidden_size
        vocab_size = gpt2_handler.vocab_size

        # Create dummy hidden state
        h = torch.randn(1, 5, hidden_size)  # (B, S, H)
        
        logits = gpt2_handler.logit_lens(h)
        
        assert logits.shape == (1, 5, vocab_size)

    def test_logit_lens_with_temperature(self, gpt2_handler):
        """Test logit_lens with temperature scaling."""
        h = torch.randn(1, 5, gpt2_handler.hidden_size)
        
        logits_t1 = gpt2_handler.logit_lens(h, temperature=1.0)
        logits_t2 = gpt2_handler.logit_lens(h, temperature=2.0)
        
        # With higher temperature, logits should be scaled down
        assert torch.allclose(logits_t1 / 2.0, logits_t2, atol=1e-5)

    def test_layerwise_distributions(self, gpt2_handler):
        """Test layerwise_distributions returns probability distributions."""
        text = "The quick brown fox"
        
        q_list = gpt2_handler.layerwise_distributions(text)
        
        # Should have num_layers + 1 distributions
        assert len(q_list) == 13
        
        vocab_size = gpt2_handler.vocab_size
        for q in q_list:
            # Each should be a probability distribution
            assert q.shape == (vocab_size,)
            assert q.sum().item() == pytest.approx(1.0, abs=1e-5)
            assert (q >= 0).all()

    def test_layerwise_distributions_position(self, gpt2_handler):
        """Test layerwise_distributions with different positions."""
        text = "Hello world test"
        
        # Last position (default)
        q_last = gpt2_handler.layerwise_distributions(text, position=-1)
        
        # First position
        q_first = gpt2_handler.layerwise_distributions(text, position=0)
        
        # Should be different distributions
        assert not torch.allclose(q_last[0], q_first[0])

    def test_layerwise_distributions_empty_text(self, gpt2_handler):
        """Test layerwise_distributions with empty text raises error."""
        with pytest.raises(ValueError, match="Empty tokenized input"):
            gpt2_handler.layerwise_distributions("")

    def test_batch_logit_lens(self, gpt2_handler):
        """Test batch_logit_lens on multiple layers."""
        text = "Test"
        tokens = gpt2_handler.tokenizer(text, return_tensors="pt")
        hidden_states = gpt2_handler.get_hidden_states(tokens["input_ids"])
        
        # Process specific layers
        logits_list = gpt2_handler.batch_logit_lens(hidden_states, layer_indices=[0, 6, 12])
        
        assert len(logits_list) == 3
        for logits in logits_list:
            assert logits.shape[-1] == gpt2_handler.vocab_size


# -----------------------------------------------------------------------------
# Logit Lens Utility Tests
# -----------------------------------------------------------------------------

class TestLogitLensUtilities:
    """Tests for standalone logit lens utilities."""

    def test_logit_lens_function(self, gpt2_handler):
        """Test the standalone logit_lens function."""
        h = torch.randn(1, 10, gpt2_handler.hidden_size)
        
        logits = logit_lens(
            h,
            gpt2_handler.get_final_norm(),
            gpt2_handler.get_lm_head(),
        )
        
        assert logits.shape == (1, 10, gpt2_handler.vocab_size)

    def test_logit_lens_probs_function(self, gpt2_handler):
        """Test logit_lens_probs returns valid probabilities."""
        h = torch.randn(1, 5, gpt2_handler.hidden_size)
        
        probs = logit_lens_probs(
            h,
            gpt2_handler.get_final_norm(),
            gpt2_handler.get_lm_head(),
        )
        
        assert probs.shape == (1, 5, gpt2_handler.vocab_size)
        # Check valid probability distribution
        assert torch.allclose(probs.sum(dim=-1), torch.ones(1, 5), atol=1e-5)
        assert (probs >= 0).all()

    def test_batch_logit_lens_function(self, gpt2_handler):
        """Test standalone batch_logit_lens function."""
        text = "Hello"
        tokens = gpt2_handler.tokenizer(text, return_tensors="pt")
        hidden_states = gpt2_handler.get_hidden_states(tokens["input_ids"])
        
        logits_list = batch_logit_lens(
            hidden_states,
            gpt2_handler.get_final_norm(),
            gpt2_handler.get_lm_head(),
            layer_indices=[0, 5, 10],
        )
        
        assert len(logits_list) == 3

    def test_layerwise_predictions_function(self, gpt2_handler):
        """Test standalone layerwise_predictions function."""
        text = "Test text"
        tokens = gpt2_handler.tokenizer(text, return_tensors="pt")
        hidden_states = gpt2_handler.get_hidden_states(tokens["input_ids"])
        
        q_list = layerwise_predictions(
            hidden_states,
            gpt2_handler.get_final_norm(),
            gpt2_handler.get_lm_head(),
            position=-1,
        )
        
        assert len(q_list) == 13  # 12 layers + embedding
        for q in q_list:
            assert q.shape == (gpt2_handler.vocab_size,)


# -----------------------------------------------------------------------------
# Edge Cases
# -----------------------------------------------------------------------------

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_token_input(self, gpt2_handler):
        """Test with single token input."""
        text = "Hi"
        q_list = gpt2_handler.layerwise_distributions(text)
        assert len(q_list) == 13

    def test_long_input(self, gpt2_handler):
        """Test with longer input."""
        text = "This is a longer test sentence with multiple tokens to process."
        q_list = gpt2_handler.layerwise_distributions(text)
        assert len(q_list) == 13

    def test_special_characters(self, gpt2_handler):
        """Test with special characters."""
        text = "Hello! @#$% 123"
        q_list = gpt2_handler.layerwise_distributions(text)
        assert len(q_list) == 13

    def test_unicode_input(self, gpt2_handler):
        """Test with unicode characters."""
        text = "Hello 世界 🌍"
        q_list = gpt2_handler.layerwise_distributions(text)
        assert len(q_list) == 13

