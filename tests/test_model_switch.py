"""Tests for mid-chat /model switching.

Covers the full model-switching stack:
- CommandDef registration (commands.py)
- Shared switch pipeline (model_switch.py)
- AIAgent.switch_model() method (run_agent.py)
- CLI handler (cli.py)
- Gateway handler (gateway/run.py)
- Edge cases and error paths
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ═══════════════════════════════════════════════════════════════════════
# CommandDef registration
# ═══════════════════════════════════════════════════════════════════════

class TestCommandRegistration:
    """Verify /model is registered correctly in the command system."""

    def test_model_command_exists(self):
        from hermes_cli.commands import COMMAND_REGISTRY
        names = [c.name for c in COMMAND_REGISTRY]
        assert "model" in names

    def test_model_command_properties(self):
        from hermes_cli.commands import COMMAND_REGISTRY
        cmd = next(c for c in COMMAND_REGISTRY if c.name == "model")
        assert cmd.category == "Configuration"
        # Available in both CLI and gateway
        assert not cmd.cli_only
        assert not cmd.gateway_only
        assert cmd.args_hint  # has usage hint

    def test_model_command_resolves(self):
        from hermes_cli.commands import resolve_command
        result = resolve_command("model")
        assert result is not None
        assert result.name == "model"

    def test_model_appears_in_help(self):
        """Verify /model shows up in the help output."""
        from hermes_cli.commands import COMMAND_REGISTRY
        config_cmds = [c for c in COMMAND_REGISTRY if c.category == "Configuration"]
        names = [c.name for c in config_cmds]
        assert "model" in names

    def test_model_in_gateway_known_commands(self):
        """Verify /model is in GATEWAY_KNOWN_COMMANDS for gateway dispatch."""
        from hermes_cli.commands import GATEWAY_KNOWN_COMMANDS
        assert "model" in GATEWAY_KNOWN_COMMANDS


# ═══════════════════════════════════════════════════════════════════════
# model_switch.py pipeline
# ═══════════════════════════════════════════════════════════════════════

class TestModelSwitchPipeline:
    """Test the shared model switch pipeline in model_switch.py."""

    def test_switch_model_result_fields(self):
        from hermes_cli.model_switch import ModelSwitchResult
        r = ModelSwitchResult(success=True, new_model="test", target_provider="openrouter")
        assert r.success is True
        assert r.new_model == "test"
        assert r.target_provider == "openrouter"
        assert r.provider_changed is False
        assert r.error_message == ""

    def test_switch_model_same_provider(self):
        """Switch model within the same provider (no provider change)."""
        from hermes_cli.model_switch import switch_model
        with patch("hermes_cli.models.parse_model_input", return_value=("openrouter", "claude-sonnet-4")), \
             patch("hermes_cli.models.detect_provider_for_model", return_value=None), \
             patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value={
                 "api_key": "test-key", "base_url": "https://openrouter.ai/api/v1", "api_mode": "",
             }), \
             patch("hermes_cli.models.validate_requested_model", return_value={
                 "accepted": True, "persist": True, "recognized": True, "message": None,
             }):
            result = switch_model("claude-sonnet-4", current_provider="openrouter")
            assert result.success is True
            assert result.new_model == "claude-sonnet-4"
            assert result.provider_changed is False

    def test_switch_model_cross_provider(self):
        """Switch to a different provider."""
        from hermes_cli.model_switch import switch_model
        with patch("hermes_cli.models.parse_model_input", return_value=("openai", "gpt-5")), \
             patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value={
                 "api_key": "sk-test", "base_url": "https://api.openai.com/v1", "api_mode": "",
             }), \
             patch("hermes_cli.models.validate_requested_model", return_value={
                 "accepted": True, "persist": True, "recognized": True, "message": None,
             }):
            result = switch_model("openai:gpt-5", current_provider="openrouter")
            assert result.success is True
            assert result.new_model == "gpt-5"
            assert result.target_provider == "openai"
            assert result.provider_changed is True

    def test_switch_model_missing_credentials(self):
        """Error when credentials can't be resolved for the target provider."""
        from hermes_cli.model_switch import switch_model
        with patch("hermes_cli.models.parse_model_input", return_value=("anthropic", "claude-opus")), \
             patch("hermes_cli.runtime_provider.resolve_runtime_provider", side_effect=Exception("No key")):
            result = switch_model("anthropic:claude-opus", current_provider="openrouter")
            assert result.success is False
            assert "credentials" in result.error_message.lower() or "no key" in result.error_message.lower()

    def test_switch_model_validation_rejected(self):
        """Error when the model is rejected by validation."""
        from hermes_cli.model_switch import switch_model
        with patch("hermes_cli.models.parse_model_input", return_value=("openrouter", "nonexistent-model")), \
             patch("hermes_cli.models.detect_provider_for_model", return_value=None), \
             patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value={
                 "api_key": "key", "base_url": "https://openrouter.ai/api/v1", "api_mode": "",
             }), \
             patch("hermes_cli.models.validate_requested_model", return_value={
                 "accepted": False, "persist": False, "recognized": False,
                 "message": "Model not found in catalog",
             }):
            result = switch_model("nonexistent-model", current_provider="openrouter")
            assert result.success is False
            assert "not found" in result.error_message.lower()

    def test_switch_to_custom_provider_success(self):
        """Bare '/model custom' with auto-detect."""
        from hermes_cli.model_switch import switch_to_custom_provider
        with patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value={
                 "base_url": "http://localhost:8080/v1", "api_key": "local-key",
             }), \
             patch("hermes_cli.runtime_provider._auto_detect_local_model", return_value="qwen2.5-72b"):
            result = switch_to_custom_provider()
            assert result.success is True
            assert result.model == "qwen2.5-72b"
            assert "localhost" in result.base_url

    def test_switch_to_custom_no_endpoint(self):
        """Error when no custom endpoint is configured."""
        from hermes_cli.model_switch import switch_to_custom_provider
        with patch("hermes_cli.runtime_provider.resolve_runtime_provider", side_effect=Exception("no endpoint")):
            result = switch_to_custom_provider()
            assert result.success is False
            assert result.error_message


# ═══════════════════════════════════════════════════════════════════════
# AIAgent.switch_model()
# ═══════════════════════════════════════════════════════════════════════

class TestAgentSwitchModel:
    """Test the AIAgent.switch_model() method."""

    def _make_agent(self, model="test-model", provider="openrouter"):
        """Create a minimal mock agent with the attributes switch_model needs."""
        from run_agent import AIAgent
        with patch.object(AIAgent, "__init__", lambda self: None):
            agent = AIAgent()
        # Set minimal required attributes
        agent.model = model
        agent.provider = provider
        agent.base_url = "https://openrouter.ai/api/v1"
        agent.api_mode = "chat_completions"
        agent.api_key = "test-key"
        agent.client = MagicMock()
        agent._client_kwargs = {"api_key": "test-key", "base_url": "https://openrouter.ai/api/v1"}
        agent._use_prompt_caching = True
        agent._cached_system_prompt = "cached prompt here"
        agent._fallback_activated = False
        agent._fallback_index = 0
        agent._anthropic_client = None
        agent._anthropic_api_key = ""
        agent._anthropic_base_url = None
        agent._is_anthropic_oauth = False
        agent._memory_store = None
        # Mock context compressor
        cc = MagicMock()
        cc.model = model
        cc.base_url = "https://openrouter.ai/api/v1"
        cc.api_key = "test-key"
        cc.provider = provider
        cc.context_length = 200000
        cc.threshold_tokens = 160000
        cc.threshold_percent = 0.8
        agent.context_compressor = cc
        # Mock _primary_runtime
        agent._primary_runtime = {
            "model": model, "provider": provider,
            "base_url": "https://openrouter.ai/api/v1",
            "api_mode": "chat_completions", "api_key": "test-key",
            "client_kwargs": dict(agent._client_kwargs),
            "use_prompt_caching": True,
            "compressor_model": model, "compressor_base_url": "https://openrouter.ai/api/v1",
            "compressor_api_key": "test-key", "compressor_provider": provider,
            "compressor_context_length": 200000, "compressor_threshold_tokens": 160000,
        }
        # Mock _create_openai_client
        agent._create_openai_client = MagicMock(return_value=MagicMock())
        # Mock _is_direct_openai_url
        agent._is_direct_openai_url = MagicMock(return_value=False)
        # Mock _invalidate_system_prompt
        agent._invalidate_system_prompt = MagicMock()
        return agent

    def test_basic_switch(self):
        """Switch from one model to another on the same provider."""
        agent = self._make_agent()
        agent.switch_model(
            new_model="claude-sonnet-4",
            new_provider="openrouter",
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
        )
        assert agent.model == "claude-sonnet-4"
        assert agent.provider == "openrouter"

    def test_system_prompt_invalidated(self):
        """System prompt must be invalidated on model switch."""
        agent = self._make_agent()
        agent.switch_model(
            new_model="claude-sonnet-4",
            new_provider="openrouter",
            api_key="key",
            base_url="https://openrouter.ai/api/v1",
        )
        agent._invalidate_system_prompt.assert_called_once()

    def test_primary_runtime_updated(self):
        """_primary_runtime must be updated (not preserved like fallback)."""
        agent = self._make_agent()
        agent.switch_model(
            new_model="gpt-5",
            new_provider="openai",
            api_key="sk-test",
            base_url="https://api.openai.com/v1",
        )
        assert agent._primary_runtime["model"] == "gpt-5"
        assert agent._primary_runtime["provider"] == "openai"

    def test_prompt_caching_reeval_claude_openrouter(self):
        """Prompt caching should be enabled for Claude on OpenRouter."""
        agent = self._make_agent()
        agent._use_prompt_caching = False
        agent.switch_model(
            new_model="anthropic/claude-sonnet-4",
            new_provider="openrouter",
            api_key="key",
            base_url="https://openrouter.ai/api/v1",
        )
        assert agent._use_prompt_caching is True

    def test_prompt_caching_reeval_non_claude(self):
        """Prompt caching should be disabled for non-Claude on OpenRouter."""
        agent = self._make_agent()
        agent._use_prompt_caching = True
        agent.switch_model(
            new_model="openai/gpt-5",
            new_provider="openrouter",
            api_key="key",
            base_url="https://openrouter.ai/api/v1",
        )
        assert agent._use_prompt_caching is False

    def test_prompt_caching_native_anthropic(self):
        """Prompt caching should be enabled for native Anthropic."""
        agent = self._make_agent()
        with patch("agent.anthropic_adapter.build_anthropic_client", return_value=MagicMock()), \
             patch("agent.anthropic_adapter.resolve_anthropic_token", return_value="sk-ant-test"), \
             patch("agent.anthropic_adapter._is_oauth_token", return_value=False):
            agent.switch_model(
                new_model="claude-sonnet-4",
                new_provider="anthropic",
                api_key="sk-ant-test",
                base_url="",
            )
        assert agent._use_prompt_caching is True
        assert agent.api_mode == "anthropic_messages"

    def test_cross_api_mode_switch(self):
        """Switching from chat_completions to anthropic_messages rebuilds client."""
        agent = self._make_agent()
        assert agent.api_mode == "chat_completions"
        with patch("agent.anthropic_adapter.build_anthropic_client", return_value=MagicMock()) as mock_build, \
             patch("agent.anthropic_adapter.resolve_anthropic_token", return_value="sk-ant-test"), \
             patch("agent.anthropic_adapter._is_oauth_token", return_value=False):
            agent.switch_model(
                new_model="claude-opus-4",
                new_provider="anthropic",
                api_key="sk-ant-test",
            )
        assert agent.api_mode == "anthropic_messages"
        assert agent.client is None  # OpenAI client cleared
        mock_build.assert_called_once()

    def test_switch_from_anthropic_to_openai(self):
        """Switching from anthropic to OpenAI-compatible clears anthropic state."""
        agent = self._make_agent()
        agent.api_mode = "anthropic_messages"
        agent._anthropic_client = MagicMock()
        agent.switch_model(
            new_model="gpt-5",
            new_provider="openai",
            api_key="sk-test",
            base_url="https://api.openai.com/v1",
        )
        assert agent.api_mode == "chat_completions"
        assert agent._anthropic_client is None
        agent._create_openai_client.assert_called()

    def test_context_compressor_updated(self):
        """Context compressor should be updated with new model's context length."""
        agent = self._make_agent()
        with patch("agent.model_metadata.get_model_context_length", return_value=128000):
            agent.switch_model(
                new_model="gpt-4o",
                new_provider="openai",
                api_key="key",
                base_url="https://api.openai.com/v1",
            )
        cc = agent.context_compressor
        assert cc.model == "gpt-4o"
        assert cc.context_length == 128000

    def test_fallback_state_reset(self):
        """Fallback state should be reset after a model switch."""
        agent = self._make_agent()
        agent._fallback_activated = True
        agent._fallback_index = 2
        agent.switch_model(
            new_model="new-model",
            new_provider="openrouter",
            api_key="key",
            base_url="https://openrouter.ai/api/v1",
        )
        assert agent._fallback_activated is False
        assert agent._fallback_index == 0

    def test_codex_api_mode_detection(self):
        """openai-codex provider should auto-detect codex_responses api_mode."""
        agent = self._make_agent()
        agent.switch_model(
            new_model="codex-mini",
            new_provider="openai-codex",
            api_key="key",
            base_url="https://api.openai.com/v1",
        )
        assert agent.api_mode == "codex_responses"


# ═══════════════════════════════════════════════════════════════════════
# CLI handler
# ═══════════════════════════════════════════════════════════════════════

class TestCLIModelSwitch:
    """Test the CLI /model handler."""

    def _make_cli(self, model="test-model", provider="openrouter"):
        """Create a minimal mock CLI instance."""
        cli = MagicMock()
        cli.model = model
        cli.provider = provider
        cli.base_url = "https://openrouter.ai/api/v1"
        cli.api_key = "test-key"
        cli.api_mode = "chat_completions"
        cli.agent = MagicMock()
        cli.agent.switch_model = MagicMock()
        # Import the real handler
        from cli import HermesCLI
        cli._handle_model_switch = HermesCLI._handle_model_switch.__get__(cli)
        return cli

    def test_no_args_shows_current(self, capsys):
        """'/model' with no args shows current model and usage."""
        cli = self._make_cli()
        with patch("hermes_cli.models._PROVIDER_LABELS", {"openrouter": "OpenRouter"}):
            cli._handle_model_switch("/model")
        captured = capsys.readouterr()
        assert "test-model" in captured.out
        assert "Usage:" in captured.out

    def test_same_model_noop(self, capsys):
        """Switching to the same model is a no-op."""
        cli = self._make_cli(model="claude-sonnet-4")
        cli._handle_model_switch("/model claude-sonnet-4")
        captured = capsys.readouterr()
        assert "Already using" in captured.out

    def test_successful_switch(self, capsys):
        """Successful model switch updates agent and CLI state."""
        cli = self._make_cli()
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.new_model = "claude-sonnet-4"
        mock_result.target_provider = "openrouter"
        mock_result.provider_changed = False
        mock_result.api_key = "key"
        mock_result.base_url = "https://openrouter.ai/api/v1"
        mock_result.api_mode = ""
        mock_result.persist = True
        mock_result.warning_message = ""
        mock_result.is_custom_target = False

        with patch("hermes_cli.model_switch.switch_model", return_value=mock_result), \
             patch("hermes_cli.models._PROVIDER_LABELS", {"openrouter": "OpenRouter"}), \
             patch("cli.save_config_value"):
            cli._handle_model_switch("/model claude-sonnet-4")

        # Agent.switch_model was called
        cli.agent.switch_model.assert_called_once()
        # CLI state was updated
        assert cli.model == "claude-sonnet-4"

    def test_failed_switch_shows_error(self, capsys):
        """Failed switch shows error message."""
        cli = self._make_cli()
        mock_result = MagicMock()
        mock_result.success = False
        mock_result.error_message = "No credentials for provider 'deepseek'"

        with patch("hermes_cli.model_switch.switch_model", return_value=mock_result):
            cli._handle_model_switch("/model deepseek:deepseek-chat")

        captured = capsys.readouterr()
        assert "Error" in captured.out
        assert "credentials" in captured.out.lower() or "deepseek" in captured.out.lower()

    def test_switch_without_agent(self, capsys):
        """Switch works even when no agent is initialized yet."""
        cli = self._make_cli()
        cli.agent = None  # No agent yet

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.new_model = "gpt-5"
        mock_result.target_provider = "openai"
        mock_result.provider_changed = True
        mock_result.api_key = "sk-test"
        mock_result.base_url = "https://api.openai.com/v1"
        mock_result.api_mode = ""
        mock_result.persist = True
        mock_result.warning_message = ""
        mock_result.is_custom_target = False

        with patch("hermes_cli.model_switch.switch_model", return_value=mock_result), \
             patch("hermes_cli.models._PROVIDER_LABELS", {"openai": "OpenAI", "openrouter": "OpenRouter"}), \
             patch("cli.save_config_value"):
            cli._handle_model_switch("/model openai:gpt-5")

        # CLI state was updated even without an agent
        assert cli.model == "gpt-5"
        assert cli.provider == "openai"

    def test_provider_changed_display(self, capsys):
        """Provider change is shown when switching cross-provider."""
        cli = self._make_cli(provider="openrouter")
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.new_model = "gpt-5"
        mock_result.target_provider = "openai"
        mock_result.provider_changed = True
        mock_result.api_key = "key"
        mock_result.base_url = "https://api.openai.com/v1"
        mock_result.api_mode = ""
        mock_result.persist = True
        mock_result.warning_message = ""
        mock_result.is_custom_target = False

        with patch("hermes_cli.model_switch.switch_model", return_value=mock_result), \
             patch("hermes_cli.models._PROVIDER_LABELS", {
                 "openrouter": "OpenRouter", "openai": "OpenAI",
             }), \
             patch("cli.save_config_value"):
            cli._handle_model_switch("/model openai:gpt-5")

        captured = capsys.readouterr()
        assert "OpenRouter" in captured.out
        assert "OpenAI" in captured.out


# ═══════════════════════════════════════════════════════════════════════
# Gateway handler
# ═══════════════════════════════════════════════════════════════════════

class TestGatewayModelSwitch:
    """Test the gateway /model handler."""

    def _make_gateway(self, tmp_path, model="test-model", provider="openrouter"):
        """Create a minimal mock gateway runner."""
        import yaml
        # Create config file
        config_dir = tmp_path / ".hermes"
        config_dir.mkdir(exist_ok=True)
        config_path = config_dir / "config.yaml"
        config = {
            "model": {
                "default": model,
                "provider": provider,
                "base_url": "https://openrouter.ai/api/v1",
            }
        }
        with open(config_path, "w") as f:
            yaml.dump(config, f)
        return config_path, config_dir

    @pytest.fixture
    def gateway_env(self, tmp_path, monkeypatch):
        config_path, config_dir = self._make_gateway(tmp_path)
        monkeypatch.setenv("HERMES_HOME", str(config_dir))
        return config_path, config_dir

    @pytest.mark.asyncio
    async def test_no_args_shows_current(self, gateway_env, monkeypatch):
        """'/model' with no args returns current model info."""
        config_path, config_dir = gateway_env
        # Patch _hermes_home in gateway module
        monkeypatch.setattr("gateway.run._hermes_home", config_dir)

        from gateway.run import GatewayRunner
        runner = MagicMock(spec=GatewayRunner)
        runner._handle_model_command = GatewayRunner._handle_model_command.__get__(runner)

        event = MagicMock()
        event.get_command_args.return_value = ""
        event.source = MagicMock()

        result = await runner._handle_model_command(event)
        assert "test-model" in result
        assert "Usage" in result or "model-name" in result

    @pytest.mark.asyncio
    async def test_successful_switch(self, gateway_env, monkeypatch):
        """Successful model switch persists to config and evicts cached agent."""
        config_path, config_dir = gateway_env
        monkeypatch.setattr("gateway.run._hermes_home", config_dir)

        from gateway.run import GatewayRunner
        runner = MagicMock(spec=GatewayRunner)
        runner._handle_model_command = GatewayRunner._handle_model_command.__get__(runner)
        runner._session_key_for_source = MagicMock(return_value="test-session")
        runner._evict_cached_agent = MagicMock()

        event = MagicMock()
        event.get_command_args.return_value = "claude-sonnet-4"
        event.source = MagicMock()

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.new_model = "claude-sonnet-4"
        mock_result.target_provider = "openrouter"
        mock_result.provider_changed = False
        mock_result.api_key = "key"
        mock_result.base_url = "https://openrouter.ai/api/v1"
        mock_result.api_mode = ""
        mock_result.persist = True
        mock_result.warning_message = ""
        mock_result.is_custom_target = False

        with patch("hermes_cli.model_switch.switch_model", return_value=mock_result), \
             patch("hermes_cli.config.save_config"):
            result = await runner._handle_model_command(event)

        assert "claude-sonnet-4" in result
        assert "switched" in result.lower() or "✅" in result
        runner._evict_cached_agent.assert_called_once_with("test-session")

    @pytest.mark.asyncio
    async def test_error_returns_message(self, gateway_env, monkeypatch):
        """Failed switch returns error message."""
        config_path, config_dir = gateway_env
        monkeypatch.setattr("gateway.run._hermes_home", config_dir)

        from gateway.run import GatewayRunner
        runner = MagicMock(spec=GatewayRunner)
        runner._handle_model_command = GatewayRunner._handle_model_command.__get__(runner)

        event = MagicMock()
        event.get_command_args.return_value = "deepseek:nonexistent"
        event.source = MagicMock()

        mock_result = MagicMock()
        mock_result.success = False
        mock_result.error_message = "Could not resolve credentials"

        with patch("hermes_cli.model_switch.switch_model", return_value=mock_result):
            result = await runner._handle_model_command(event)

        assert "❌" in result
        assert "credentials" in result.lower()

    @pytest.mark.asyncio
    async def test_same_model_noop(self, gateway_env, monkeypatch):
        """Same model returns 'already using' message."""
        config_path, config_dir = gateway_env
        monkeypatch.setattr("gateway.run._hermes_home", config_dir)

        from gateway.run import GatewayRunner
        runner = MagicMock(spec=GatewayRunner)
        runner._handle_model_command = GatewayRunner._handle_model_command.__get__(runner)

        event = MagicMock()
        event.get_command_args.return_value = "test-model"
        event.source = MagicMock()

        result = await runner._handle_model_command(event)
        assert "Already using" in result


# ═══════════════════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Edge cases and integration scenarios."""

    def test_custom_auto_result_fields(self):
        from hermes_cli.model_switch import CustomAutoResult
        r = CustomAutoResult(success=True, model="llama-3.3", base_url="http://localhost:11434")
        assert r.success
        assert r.model == "llama-3.3"

    def test_switch_model_custom_endpoint(self):
        """'/model custom:model-name' sets is_custom_target."""
        from hermes_cli.model_switch import switch_model
        with patch("hermes_cli.models.parse_model_input", return_value=("custom", "my-model")), \
             patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value={
                 "api_key": "local-key", "base_url": "http://localhost:8080/v1", "api_mode": "",
             }), \
             patch("hermes_cli.models.validate_requested_model", return_value={
                 "accepted": True, "persist": True, "recognized": False, "message": None,
             }):
            result = switch_model("custom:my-model", current_provider="openrouter")
            assert result.success is True
            assert result.is_custom_target is True

    def test_opencode_api_mode_recompute(self):
        """OpenCode providers recompute api_mode for the new model."""
        from hermes_cli.model_switch import switch_model
        with patch("hermes_cli.models.parse_model_input", return_value=("opencode-zen", "claude-opus")), \
             patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value={
                 "api_key": "key", "base_url": "https://example.com", "api_mode": "chat_completions",
             }), \
             patch("hermes_cli.models.validate_requested_model", return_value={
                 "accepted": True, "persist": True, "recognized": True, "message": None,
             }), \
             patch("hermes_cli.models.opencode_model_api_mode", return_value="anthropic_messages") as mock_oc:
            result = switch_model("opencode-zen:claude-opus", current_provider="openrouter")
            assert result.success is True
            assert result.api_mode == "anthropic_messages"
            mock_oc.assert_called_with("opencode-zen", "claude-opus")

    def test_provider_label_in_result(self):
        """Result includes human-readable provider label."""
        from hermes_cli.model_switch import switch_model
        with patch("hermes_cli.models.parse_model_input", return_value=("openrouter", "test")), \
             patch("hermes_cli.models.detect_provider_for_model", return_value=None), \
             patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value={
                 "api_key": "key", "base_url": "https://openrouter.ai/api/v1", "api_mode": "",
             }), \
             patch("hermes_cli.models.validate_requested_model", return_value={
                 "accepted": True, "persist": True, "recognized": True, "message": None,
             }), \
             patch("hermes_cli.models._PROVIDER_LABELS", {"openrouter": "OpenRouter"}):
            result = switch_model("test", current_provider="openrouter")
            assert result.provider_label == "OpenRouter"
