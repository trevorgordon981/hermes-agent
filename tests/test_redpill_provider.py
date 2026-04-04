# =============================================================================
# RedPill TEE-protected provider tests
# =============================================================================

class TestRedPillProviderConfiguration:
    """Verify RedPill provider is correctly configured."""

    def test_redpill_provider_exists_in_auth(self):
        """RedPill should be registered in PROVIDER_REGISTRY."""
        from hermes_cli.auth import PROVIDER_REGISTRY
        
        assert "redpill" in PROVIDER_REGISTRY
        config = PROVIDER_REGISTRY["redpill"]
        
        assert config.id == "redpill"
        assert config.name == "RedPill"
        assert config.auth_type == "api_key"
        assert config.inference_base_url == "https://api.redpill.ai/v1"
        assert config.api_key_env_vars == ("REDPILL_API_KEY",)
        assert config.base_url_env_var == "REDPILL_BASE_URL"

    def test_redpill_provider_exists_in_models(self):
        """RedPill model list should be defined."""
        from hermes_cli.models import _PROVIDER_MODELS
        
        assert "redpill" in _PROVIDER_MODELS
        models = _PROVIDER_MODELS["redpill"]
        
        # Should have at least 15 curated models
        assert len(models) >= 15, f"Expected 15+ RedPill models, got {len(models)}"

    def test_redpill_provider_exists_in_main(self):
        """RedPill model list in main.py should match models.py."""
        from hermes_cli.main import _PROVIDER_MODELS as main_models
        from hermes_cli.models import _PROVIDER_MODELS as models_models
        
        assert "redpill" in main_models
        assert "redpill" in models_models
        assert main_models["redpill"] == models_models["redpill"], \
            "RedPill model lists must be identical in main.py and models.py"

    def test_redpill_model_list_contains_key_models(self):
        """Verify RedPill list includes important models."""
        from hermes_cli.models import _PROVIDER_MODELS
        
        models = _PROVIDER_MODELS["redpill"]
        
        # Flagship models
        assert "qwen/qwen3.5-27b" in models
        assert "qwen/qwen3.5-397b-a17b" in models
        
        # Strong reasoning
        assert "z-ai/glm-4.7" in models
        assert "deepseek/deepseek-v3.2" in models
        
        # Budget options
        assert "qwen/qwen-2.5-7b-instruct" in models
        assert "openai/gpt-oss-20b" in models
        
        # Vision capability
        assert "qwen/qwen3-vl-30b-a3b-instruct" in models

    def test_redpill_aliases_resolve(self):
        """RedPill aliases should resolve correctly."""
        from hermes_cli.auth import resolve_provider
        
        assert resolve_provider("redpill") == "redpill"
        assert resolve_provider("redpill-ai") == "redpill"
        assert resolve_provider("red-pill") == "redpill"

    def test_redpill_in_cli_choices(self):
        """RedPill should appear in CLI chat --provider choices."""
        import re
        
        # Read main.py source to verify redpill is in chat parser choices
        with open('hermes_cli/main.py', 'r') as f:
            source = f.read()
        
        # Find the chat_parser.add_argument("--provider", choices=[...]) line
        chat_provider_pattern = r'chat_parser\.add_argument\s*\(\s*["\']--provider["\'].*?choices=\[([^\]]+)\]'
        match = re.search(chat_provider_pattern, source, re.DOTALL)
        
        assert match, "Could not find chat_parser --provider choices definition"
        
        choices_str = match.group(1)
        assert 'redpill' in choices_str, \
            f"RedPill should be in chat --provider choices: {choices_str}"


class TestRedPillModelCapabilities:
    """Verify RedPill model list has proper capability annotations."""

    def test_redpill_models_have_capability_comments(self):
        """RedPill model list should include capability comments."""
        from hermes_cli.models import _PROVIDER_MODELS
        
        # Read the source to check for comments
        import inspect
        from hermes_cli import models
        
        source = inspect.getsource(models)
        
        # Check for capability annotation pattern
        assert "# tools:" in source or "# tools:Y" in source, \
            "RedPill models should have capability annotations (tools/vision/context)"

    def test_redpill_vision_model_flagged(self):
        """Vision-capable model should be identifiable."""
        from hermes_cli.models import _PROVIDER_MODELS
        
        models = _PROVIDER_MODELS["redpill"]
        
        # At least one vision model should be present
        vision_models = [m for m in models if "vl" in m.lower()]
        assert len(vision_models) >= 1, \
            "RedPill should offer at least one vision-capable model"
